function merge_particles!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        merge_particles!(system, v_ode, u_ode, semi)
    end

    return semi
end

@inline merge_particles!(system, v_ode, u_ode, semi) = system

@inline function merge_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return merge_particles!(system, system.particle_refinement, v_ode, u_ode, semi)
end

@inline merge_particles!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode,
                         semi) = system

@inline function merge_particles!(system::AbstractFluidSystem, refinement, v_ode, u_ode,
                                  semi; merge_iter=3)
    (; buffer, cache) = system
    (; delete_candidates) = cache

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Reset delete candidates 
    @threaded semi for particle in eachindex(delete_candidates)
        delete_candidates[particle] = false
    end

    # Merge the particles
    for _ in 1:merge_iter
        # TrixiParticles.@autoinfiltrate
        collect_merge_candidates!(system, refinement, v, u, semi)
        # TrixiParticles.@autoinfiltrate
        apply_merging!(system, refinement, v, u, semi)
        # TrixiParticles.@autoinfiltrate

        # Update the nhs
        update_system_buffer!(buffer)
        neighborhood_search = get_neighborhood_search(system, semi)
        update_nhs!(neighborhood_search, system, system, u, u, semi)
        # update_nhs(semi, system, system) # Quick fix

        # TrixiParticles.@autoinfiltrate
    end

    # TrixiParticles.@autoinfiltrate

    return system
end

function collect_merge_candidates!(system::AbstractFluidSystem, refinement, v, u, semi)
    (; spacing_ratio) = refinement
    (; reference_mass, merge_candidates, delete_candidates) = system.cache

    set_zero!(merge_candidates)
    system_coords = current_coordinates(u, system)

    # Collect merge candidates
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Do not merge a particle that was deleted
        delete_candidates[particle] && return

        # neighbor == 26 && TrixiParticles.@autoinfiltrate
        delete_candidates[neighbor] && return

        particle == neighbor && return

        # Do not merge particle if distance >= (h_a + h_b) / 2
        h_a = smoothing_length(system, particle)
        h_b = smoothing_length(system, neighbor)
        distance >= (h_a + h_b) / 2 && return

        # Do not merge particles if m_a + m_b >= max(m_max_a, m_max_b)
        m_merge = hydrodynamic_mass(system, particle) + hydrodynamic_mass(system, neighbor)
        m_merge >=
        spacing_ratio * max(reference_mass[particle], reference_mass[neighbor]) && return

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

function apply_merging!(system::AbstractFluidSystem, refinement, v, u, semi)
    (; smoothing_kernel, cache) = system
    (; reference_mass, merge_candidates, delete_candidates) = cache

    ELTYPE = eltype(u)
    NDIMS = ndims(system)

    inv_ndims = one(ELTYPE) / NDIMS
    kernel_0_1 = kernel(smoothing_kernel, zero(ELTYPE), one(ELTYPE))

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

            vel_a = current_velocity(v, system, particle)
            vel_b = current_velocity(v, system, candidate)

            pos_merge = (m_a * pos_a + m_b * pos_b) / m_merge
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

            # Update reference mass
            reference_mass[particle] += reference_mass[candidate]

            # Let this thread mark the other particle as deleted 
            # and deactivate it to avoid race condition 
            delete_candidates[candidate] = true
            deactivate_particle!(system, candidate, v, u)
        end
    end

    return system
end
