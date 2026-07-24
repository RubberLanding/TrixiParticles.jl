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

@inline merge_particles!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode, semi) = system

@inline function merge_particles!(system::AbstractFluidSystem, refinement, v_ode, u_ode, semi; merge_iter=3)
    (; delete_candidates) = refinement
    (; n_delete_particles) = refinement.resize_buffer
    (; candidate_flags) = system.cache

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Reset delete candidates 
    @threaded semi for particle in eachindex(delete_candidates)
        delete_candidates[particle] = false
    end

    # Merge the particles
    for _ in 1:merge_iter
        collect_merge_candidates!(system, refinement, v, u, semi)
        apply_merging!(system, refinement, v, u, semi)
    end
    
    # Update the counter for the particles to delete
    @threaded semi for particle in eachparticle(system)
        candidate_flags[particle] = delete_candidates[particle] ? 1 : 0
    end 
    fill!(n_delete_particles, sum(candidate_flags))

    return system 
end 

function collect_merge_candidates!(system::AbstractFluidSystem, refinement, v, u, semi)
    (; spacing_ratio, split_candidates, merge_candidates, delete_candidates) = refinement
    (; reference_mass) = system.cache 

    set_zero!(merge_candidates)
    system_coords = current_coordinates(u, system)

    # Collect merge candidates
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Do not merge a particle that was split
        split_candidates[particle] && return
        split_candidates[neighbor] && return

        # Do not merge a particle that was deleted
        delete_candidates[particle] && return
        delete_candidates[neighbor] && return

        particle == neighbor && return

        m_a = hydrodynamic_mass(system, particle)
        m_b = hydrodynamic_mass(system, neighbor)
        m_max = spacing_ratio * reference_mass[particle]
        m_a > m_max && return 

        m_merge = m_a + m_b
        m_max_min = min(m_max, spacing_ratio * reference_mass[neighbor])
        m_merge >= m_max_min && return 
        
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
    (; merge_candidates, delete_candidates) = refinement
    (; reference_mass) = cache 

    ELTYPE = eltype(u)
    NDIMS = ndims(system)

    inv_ndims = one(ELTYPE) / NDIMS
    kernel_0_1 = kernel(smoothing_kernel, zero(ELTYPE), one(ELTYPE))

    delete_mass = zero(ELTYPE)
    delete_velocity = zero(SVector{NDIMS, ELTYPE})
    delete_position = SVector{NDIMS, ELTYPE}(ntuple(_ -> typemax(ELTYPE), NDIMS))

    # Merge and delete particles
    @threaded semi for particle in eachindex(merge_candidates)
        candidate = merge_candidates[particle]

        delete_candidates[particle] && return 
        candidate == 0 && return
        particle != merge_candidates[candidate] && return

        if particle < candidate
            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(system, candidate)

            m_merge = m_a + m_b

            pos_a = current_coords(u, system, particle)
            pos_b = current_coords(u, system, candidate)

            vel_a = current_velocity(v, system, particle)
            vel_b = current_velocity(v, system, candidate)

            pos_merge = (m_a * pos_a + m_b * pos_b) / m_merge
            vel_merge = (m_a * vel_a + m_b * vel_b) / m_merge

            # Update position and velocity
            set_particle_position!(u, system, particle, pos_merge) # (Eq. 32)
            set_particle_velocity!(v, system, particle, vel_merge) # (Eq. 33)

            # Update smoothing length 
            h_a = smoothing_length(system, particle)
            h_b = smoothing_length(system, candidate)
            tmp_m = m_merge * kernel_0_1 
            tmp_a = m_a * kernel(smoothing_kernel, norm(pos_merge - pos_a), h_a)
            tmp_b = m_b * kernel(smoothing_kernel, norm(pos_merge - pos_b), h_b)
            smoothing_length_merge = (tmp_m / (tmp_a + tmp_b))^inv_ndims

            set_particle_smoothing_length!(system, particle, smoothing_length_merge) # (Eq. 34)

            # Update mass
            set_particle_mass!(system, particle, m_merge)

            # Update reference mass
            reference_mass[particle] += reference_mass[candidate]

        else
            # Disable the particle to be deleted
            delete_candidates[particle] = true
            set_particle_mass!(system, particle, delete_mass)
            set_particle_velocity!(v, system, particle, delete_velocity)
            set_particle_position!(u, system, particle, delete_position)
        end
    end

    return system
end

