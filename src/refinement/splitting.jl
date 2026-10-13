function split_particles!(semi, v_ode, u_ode, _v_ode, _u_ode)
    resize = false

    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return
        (; buffer, particle_refinement, cache) = system
        (; active_particle, active_particle_count) = buffer

        n_new_particles = collect_split_candidates!(system, particle_refinement, v_ode,
                                                    u_ode, semi)
        # TrixiParticles.@autoinfiltrate
        current_capacity = length(active_particle)
        required_capacity = active_particle_count[] + n_new_particles

        # TrixiParticles.@autoinfiltrate

        if required_capacity > 0.95 * current_capacity
            resize = true
            new_capacity = max(round(Int, 1.25 * current_capacity),
                               required_capacity + buffer.buffer_size)

            resize!(system, new_capacity)
        end
    end

    # TrixiParticles.@autoinfiltrate

    if resize

        # Check for each OpenBoundarySystem whether
        # its fluid system got resized. If so, resize the
        # boundary system as well
        foreach_system(semi) do open_system
            !isa(open_system, OpenBoundarySystem) && return
            fluid_capacity = nparticles(open_system.fluid_system)
            open_capacity = nparticles(open_system)

            if fluid_capacity > open_capacity
                resize!(open_system.fluid_candidates, fluid_capacity)
            end
        end

        # Global resize of the semidiscretization 
        resize!(v_ode, u_ode, _v_ode, _u_ode, semi)
    end

    # TrixiParticles.@autoinfiltrate

    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return
        (; buffer, particle_refinement, cache) = system

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        # We just expanded the global v_ode and u_ode arrays with `resize!`, 
        # meaning the tails of these arrays are filled with garbage data
        # We overwrite this data by explictly deactivating all inactive particles
        capacity = nparticles(system)
        for particle in (buffer.active_particle_count[] + 1):capacity
            TrixiParticles.deactivate_particle!(system, particle, v, u)

            # TODO
            # Initialize pressure/density rows if they exist in EDAC
            for dim in (ndims(system) + 1):size(v, 1)
                v[dim, particle] = 0.0
            end
        end

        apply_splitting!(system, particle_refinement, v, u, semi)

        # Update the nhs for the fluid-fluid self-interaction, 
        # so the following merge finds the new particles
        update_system_buffer!(buffer)
        nhs = get_neighborhood_search(system, semi)
        update_nhs!(nhs, system, system, u, u, semi)
    end

    # TrixiParticles.@autoinfiltrate

    return semi
end

@inline function collect_split_candidates!(system::AbstractFluidSystem, refinement, v_ode,
                                           u_ode,
                                           semi)
    (; splitting_pattern) = refinement
    (; reference_mass, split_candidates, delete_candidates) = system.cache

    split_candidates .= false
    @threaded semi for particle in eachparticle(system)
        is_alive = !delete_candidates[particle]
        particle_mass = hydrodynamic_mass(system, particle)
        particle_mass_max = 1.05 * reference_mass[particle]

        split_candidates[particle] = is_alive && (particle_mass > particle_mass_max)
    end

    # Explicitly check whether a split candidate will create 
    # children inside a boundary or solid 
    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)
    backend=semi.parallelization_backend

    @threaded semi for particle in eachindex(split_candidates)
        !split_candidates[particle] && return
        h = smoothing_length(system, particle)
        split_radius = h * refinement.splitting_pattern.epsilon

        # Loop over all systems in the semidiscretization
        foreach_system(semi) do neighbor_system
            # Skip checking fluid systems against themselves (we only care about boundaries)
            !isa(neighbor_system, Union{AbstractBoundarySystem, AbstractStructureSystem}) &&
                return
            min_distance = split_radius +
                           (neighbor_system.initial_condition.particle_spacing * 0.5)

            u_neighbor = wrap_u(u_ode, neighbor_system, semi)
            neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
            nhs = get_neighborhood_search(system, neighbor_system, semi)
            foreach_neighbor(system_coords, neighbor_coords, nhs, backend,
                             particle) do particle, neighbor, pos_diff, distance

                # If the split candidate is too close to 
                # any boundary or solid, stop the split
                if distance < min_distance
                    split_candidates[particle] = false
                end
            end
        end
    end

    n_childs_exclude_center = nchilds(splitting_pattern) - 1
    total_new_particles = sum(split_candidates) * n_childs_exclude_center

    return total_new_particles
end

@inline function apply_splitting!(system::AbstractFluidSystem, particle_refinement, v, u,
                                  semi)
    (; splitting_pattern) = particle_refinement
    (; candidate_flags, candidate_offsets, reference_mass, split_candidates) = system.cache
    (; alpha, relative_position) = splitting_pattern
    (; active_particle, active_particle_count) = system.buffer

    @threaded semi for particle in eachindex(split_candidates)
        candidate_flags[particle] = split_candidates[particle] ? 1 : 0
    end

    cumsum!(candidate_offsets, candidate_flags)

    n_childs_exclude_center = nchilds(splitting_pattern) - 1
    n_childs = nchilds(splitting_pattern)
    buffer_start_idx = active_particle_count[]

    @threaded semi for particle in eachparticle(system)
        !split_candidates[particle] && return

        smoothing_length_old = smoothing_length(system, particle)
        smoothing_length_new = smoothing_length_old * alpha
        set_particle_smoothing_length!(system, particle, smoothing_length_new)

        mass_old = hydrodynamic_mass(system, particle)
        mass_new = mass_old / n_childs
        set_particle_mass!(system, particle, mass_new)

        rho_particle = particle_refinement.reference_density
        # rho_particle = current_density(v, system, particle) # Not reliable for EDAC

        reference_mass_particle = reference_mass[particle]
        pos_particle = current_coords(u, system, particle)
        vel_particle = current_velocity(v, system, particle)

        pressure_particle = current_pressure(v, system, particle)

        particle_offset = (candidate_offsets[particle] - 1) * n_childs_exclude_center
        for child_id_local in 1:n_childs_exclude_center
            global_child_idx = buffer_start_idx + particle_offset + child_id_local

            rel_pos = smoothing_length_old * relative_position[child_id_local]
            new_pos = pos_particle + rel_pos

            global_child_idx > 10837 && TrixiParticles.@autoinfiltrate

            set_particle_mass!(system, global_child_idx, mass_new)
            set_particle_smoothing_length!(system, global_child_idx, smoothing_length_new)
            set_particle_density!(v, system, global_child_idx, rho_particle)
            set_particle_velocity!(v, system, global_child_idx, vel_particle)
            set_particle_position!(u, system, global_child_idx, new_pos)
            set_particle_pressure!(v, system, global_child_idx, pressure_particle)

            reference_mass[global_child_idx] = reference_mass_particle
            active_particle[global_child_idx] = true
        end
    end

    return system
end
