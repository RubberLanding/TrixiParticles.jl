function merge_particles!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        merge_particles!(system, v_ode, u_ode, semi)
    end

    return semi
end

@inline merge_particles!(system, v_ode, u_ode, semi; kwargs...) = system

@inline function merge_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi;
                                  kwargs...)
    return merge_particles!(system, system.particle_refinement, v_ode, u_ode, semi;
                            kwargs...)
end

@inline merge_particles!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode,
                         semi; kwargs...) = system

# Merge before splitting. As in the reference code, this is done once and every particle
# that is not split is a candidate, since its mass is below `m_max`.
@inline merge_particles_pre_split!(system, v_ode, u_ode, semi) = system

@inline function merge_particles_pre_split!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    refinement = system.particle_refinement
    isnothing(refinement) && return system

    return merge_particles!(system, refinement, v_ode, u_ode, semi;
                            merge_iter=1, mass_gate_factor=refinement.max_mass_factor)
end

# By default, merge after splitting. As in the reference code, this is done three times
# and only particles below `m_min` are candidates, so that the children of a split
# are merged back only if they are too small for the target spacing.
@inline function merge_particles!(system::AbstractFluidSystem, refinement, v_ode, u_ode,
                                  semi; merge_iter=3,
                                  mass_gate_factor=refinement.min_mass_factor)
    (; cache) = system
    (; delete_candidates) = cache

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Reset delete candidates 
    @threaded semi for particle in eachindex(delete_candidates)
        delete_candidates[particle] = false
    end

    # TrixiParticles.@autoinfiltrate

    # Merge the particles
    for i in 1:merge_iter
        # TrixiParticles.@autoinfiltrate
        collect_merge_candidates!(system, refinement, v, u, semi, mass_gate_factor)
        # TrixiParticles.@autoinfiltrate
        apply_merging!(system, refinement, v, u, semi)
        # TrixiParticles.@autoinfiltrate

        # Merging creates holes that we need to fix by compacting the system
        # so all active particles are at the beginning
        compact_system!(system, v_ode, u_ode, semi)
        neighborhood_search = get_neighborhood_search(system, semi)
        update_nhs!(neighborhood_search, system, system, u, u, semi)
        # TrixiParticles.@autoinfiltrate
    end

    return system
end

function collect_merge_candidates!(system::AbstractFluidSystem, refinement, v, u, semi,
                                   mass_gate_factor)
    (; max_mass_factor) = refinement
    (; reference_mass, merge_candidates, delete_candidates) = system.cache

    set_zero!(merge_candidates)
    system_coords = current_coordinates(u, system)
    neighborhood_search = get_neighborhood_search(system, semi)
    backend = semi.parallelization_backend

    # Collect merge candidates
    @threaded semi for particle in eachparticle(system)
        particle_radius = compact_support(system, system, particle)
        @inbounds foreach_neighbor(system_coords, system_coords,
                                   neighborhood_search, backend,
                                   particle;
                                   search_radius=particle_radius) do particle, neighbor,
                                                                     pos_diff, distance
            # Do not merge a particle that was deleted
            delete_candidates[particle] && return
            delete_candidates[neighbor] && return
            particle == neighbor && return

            # Do not merge particle if distance >= (h_a + h_b) / 2
            h_a = smoothing_length(system, particle)
            h_b = smoothing_length(system, neighbor)
            distance >= (h_a + h_b) / 2 && return

            # Only merge particles below the mass gate.
            # Before splitting, this is `m_max`, after splitting, this is `m_min`.
            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(system, neighbor)
            m_a >= mass_gate_factor * reference_mass[particle] && return
            m_b >= mass_gate_factor * reference_mass[neighbor] && return

            # Do not merge particles if m_a + m_b >= min(m_max_a, m_max_b)
            m_a + m_b >=
            max_mass_factor * min(reference_mass[particle], reference_mass[neighbor]) &&
                return

            # TrixiParticles.@autoinfiltrate

            if merge_candidates[particle] == 0
                merge_candidates[particle] = neighbor
            else
                stored_neighbor = current_coords(u, system, merge_candidates[particle])
                pos_diff_stored = stored_neighbor - current_coords(u, system, particle)

                if distance < norm(pos_diff_stored)
                    merge_candidates[particle] = neighbor
                end
            end
        end
    end
end

function apply_merging!(system::AbstractFluidSystem, refinement, v, u, semi)
    (; smoothing_kernel, cache) = system
    (; merge_candidates, delete_candidates) = cache

    ELTYPE = eltype(u)
    NDIMS = ndims(system)

    inv_ndims = one(ELTYPE) / NDIMS
    kernel_0_1 = kernel(smoothing_kernel, zero(ELTYPE), one(ELTYPE))
    periodic_box = get_neighborhood_search(system, semi).periodic_box

    # Merge and delete particles
    @threaded semi for particle in eachparticle(system)
        candidate = merge_candidates[particle]

        delete_candidates[particle] && return
        candidate == 0 && return
        particle != merge_candidates[candidate] && return

        if particle < candidate
            # TrixiParticles.@autoinfiltrate
            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(system, candidate)

            m_merge = m_a + m_b

            pos_a = current_coords(u, system, particle)
            pos_b = current_coords(u, system, candidate)

            # Compute the shortest distance between both points to correctly calculate 
            # the merged position. Without unwrapping, particles on opposite periodic boundaries 
            # would incorrectly average out and merge into the center of the domain.            
            pos_diff = shortest_periodic_diff(pos_a, pos_b, periodic_box)
            pos_b_unwrapped = pos_a + pos_diff
            pos_merge_unwrapped = (m_a * pos_a + m_b * pos_b_unwrapped) / m_merge
            pos_merge = PointNeighbors.periodic_coords(pos_merge_unwrapped, periodic_box)

            vel_a = current_velocity(v, system, particle)
            vel_b = current_velocity(v, system, candidate)
            vel_merge = (m_a * vel_a + m_b * vel_b) / m_merge

            # Update position and velocity (Eq. 32 and 33)
            set_particle_position!(u, system, particle, pos_merge)
            set_particle_velocity!(v, system, particle, vel_merge)

            # Update smoothing length (Eq. 34)
            h_a = smoothing_length(system, particle)
            h_b = smoothing_length(system, candidate)
            tmp_m = m_merge * kernel_0_1
            tmp_a = m_a * kernel(smoothing_kernel, norm(pos_merge - pos_a), h_a)
            tmp_b = m_b * kernel(smoothing_kernel, norm(pos_merge - pos_b), h_b)
            smoothing_length_merge = (tmp_m / (tmp_a + tmp_b))^inv_ndims
            set_particle_smoothing_length!(system, particle, smoothing_length_merge)

            # Update mass
            set_particle_mass!(system, particle, m_merge)

            # Let this thread mark the other particle as deleted 
            # and deactivate it to avoid race condition 
            delete_candidates[candidate] = true
            deactivate_particle!(system, candidate, v, u)
        end
    end

    return system
end

@inline shortest_periodic_diff(pos_a, pos_b, ::Nothing) = pos_b - pos_a

# Compute the shortest difference vector between points a and b
@inline function shortest_periodic_diff(pos_a, pos_b, periodic_box)
    domain_size = periodic_box.size
    diff = pos_b - pos_a
    return diff .- domain_size .* round.(diff ./ domain_size)
end
