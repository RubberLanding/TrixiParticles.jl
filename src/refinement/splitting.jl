function split_particles!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        split_particles!(system, v_ode, u_ode, semi)
    end
    return semi
end

@inline split_particles!(system, v_ode, u_ode, semi) = system 

function split_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return split_particles!(system, system.particle_refinement, v_ode, u_ode, semi)
end 

@inline split_particles!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode, semi) = system 

function split_particles!(system::AbstractFluidSystem, refinement, v_ode, u_ode, semi)
    (; resize_buffer) = refinement 
    (; n_add_particles) = resize_buffer

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Look for particles to split flag them
    _n_add_particles = collect_split_candidates!(system, refinement, v, u, semi)
    
    # With the current resizing approach, `n_add_particles` gets set once 
    # per refinement, at this location. Thus we do not need to increment. 
    # _n_add_particles += n_add_particles[1]

    # Store the total number of additional particle in the resize buffer
    fill!(n_add_particles, _n_add_particles)

    # Resize the buffer to hold additional particle data
    resize_buffer!(resize_buffer, system, _n_add_particles)

    if _n_add_particles <= 0 
        return system 
    end

    # Update data for parent and child split particles
    apply_splitting!(system, refinement, v, u, semi)
end 

@inline function collect_split_candidates!(system::AbstractFluidSystem, refinement, v, u, semi)
    (; max_spacing_ratio, split_candidates, delete_candidates, 
    splitting_pattern) = refinement
    (; reference_mass) = system.cache

        @threaded semi for particle in eachparticle(system)
        is_alive = !delete_candidates[particle]
        particle_mass = hydrodynamic_mass(system, particle)
        particle_mass_max = max_spacing_ratio * reference_mass[particle]

        split_candidates[particle] = is_alive && (particle_mass > particle_mass_max)
    end

    n_childs_exclude_center = nchilds(system, splitting_pattern) - 1
    total_new_particles = sum(split_candidates) * n_childs_exclude_center 

    return total_new_particles
end

@inline function apply_splitting!(system::AbstractFluidSystem, particle_refinement, v, u, semi)
    (; split_candidates, candidate_flags, candidate_offsets, splitting_pattern, resize_buffer) = particle_refinement
    (; alpha, relative_position) = splitting_pattern
    (; masses_new, densities_new, smoothing_lengths_new, velocities_new, positions_new) = resize_buffer

    @threaded semi for particle in eachparticle(system)
        candidate_flags[particle] = split_candidates[particle] ? 1 : 0
    end 

    cumsum!(candidate_offsets, candidate_flags)

    n_childs_exclude_center = nchilds(system, splitting_pattern) - 1
    NDIMS = ndims(system)
    @threaded semi for particle in eachindex(split_candidates)
        !split_candidates[particle] && return
        smoothing_length_old = smoothing_length(system, particle)
        mass_old = hydrodynamic_mass(system, particle)
        mass_new = mass_old / nchilds(system, splitting_pattern)
        rho_a = current_density(v, system, particle)
        pos_center = current_coords(u, system, particle)
        vel_center = current_velocity(v, system, particle)
        smoothing_length_new = smoothing_length_old * alpha

        set_particle_mass!(system, particle, mass_new)
        set_particle_smoothing_length!(system, particle, smoothing_length_new)

        particle_offset = (candidate_offsets[particle] - 1) * n_childs_exclude_center
        for child_id_local in 1:n_childs_exclude_center
            child = particle_offset + child_id_local
            rel_pos = smoothing_length_old * relative_position[child_id_local]
            new_pos = pos_center + rel_pos

            masses_new[child] = mass_new

            densities_new[child] = rho_a

            smoothing_lengths_new[child] = smoothing_length_new

            child_offset = (child - 1) * NDIMS
            for dim in 1:NDIMS
                linear_idx = child_offset + dim
                velocities_new[linear_idx] = vel_center[dim]
                positions_new[linear_idx] = new_pos[dim]
            end
        end
    end

    return system
end
