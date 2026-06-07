function merge_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        merge_particles!(system, semi, v, u)
    end

    deleteat!(semi, v_ode, u_ode, v_tmp, u_tmp)

    return semi
end

@inline merge_particles!(system, semi, v, u) = system

@inline function merge_particles!(system::AbstractFluidSystem, semi, v, u)
    return merge_particles!(system, system.particle_refinement, semi, v, u)
end

@inline merge_particles!(system::AbstractFluidSystem, ::Nothing, semi, v, u) = system

@inline function merge_particles!(system::AbstractFluidSystem, particle_refinement, semi, v, u)
    (; delete_candidates, smoothing_length_factor, neighbor_mass_sum, neighbor_count ) = system.particle_refinement
    ELTYPE = eltype(u)
    inv_density = one(ELTYPE) / system.state_equation.reference_density
    inv_ndims  = one(ELTYPE) / ndims(system)

    @threaded semi for particle in eachindex(delete_candidates)
        delete_candidates[particle] = false
    end

    # Merge particles iteratively
    for _ in 1:3
        merge_particles_inner!(system, particle_refinement, semi, v, u)
    end

    # TODO
    # deleteat!(semi, ...)

    neighborhood_search = get_neighborhood_search(system, semi)
    system_coords = current_coordinates(u, system)
    PointNeighbors.update!(neighborhood_search, system_coords, system_coords)

    set_zero!(neighbor_mass_sum)
    set_zero!(neighbor_count)

    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        delete_candidates[particle] && return 
        delete_candidates[neighbor] && return 

        neighbor_mass_sum[particle] += hydrodynamic_mass(system, neighbor)
        neighbor_count[particle] += 1
    end 

    @threaded semi for particle in eachindex(neighbor_mass_sum)
        if delete_candidates[particle] || neighbor_count[particle] == 0
            continue
        end

        avg_mass = neighbor_mass_sum[particle] * (one(ELTYPE) / neighbor_count[particle])
        system.smoothing_length[particle] = smoothing_length_factor * (inv_density * avg_mass)^inv_ndims
    end

    return system
end

function merge_particles_inner!(system, particle_refinement, semi, v, u)
    (; smoothing_kernel, cache) = system
    (; max_spacing_ratio, merge_candidates, delete_candidates) = particle_refinement
    (; reference_mass) = cache 

    ELTYPE = eltype(u)
    NDIMS = ndims(system)

    set_zero!(merge_candidates)
    system_coords = current_coordinates(u, system)

    # Collect merge candidates
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        delete_candidates[particle] && return
        delete_candidates[neighbor] && return
        particle == neighbor && return

        m_a = hydrodynamic_mass(system, particle)
        m_b = hydrodynamic_mass(system, neighbor)
        m_max = max_spacing_ratio * reference_mass[particle]

        if m_a <= m_max
            m_merge = m_a + m_b
            m_max_min = min(m_max, max_spacing_ratio * reference_mass[neighbor])
            if m_merge < m_max_min
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

    inv_ndims = one(ELTYPE) / NDIMS
    kernel_0_1 = kernel(smoothing_kernel, zero(ELTYPE), one(ELTYPE))

    # Merge and delete particles
    @threaded semi for particle in eachindex(merge_candidates)
        candidate = merge_candidates[particle]

        delete_candidates[particle] && continue 
        delete_candidates[candidate] && continue 

        if candidate != 0
            if particle == merge_candidates[candidate]
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
                    set_particle_velocity!(v, system, particle, vel_merge)
                    set_particle_position!(u, system, particle, pos_merge)

                    # Update smoothing length 
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
                    reference_mass[particle] += reference_mass[particle] + reference_mass[candidate]

                else
                    # Disable the particle to be deleted
                    delete_candidates[particle] = true
                    set_particle_mass!(system, candidate, zero(ELTYPE))
                    set_particle_velocity(v, system, candidate, zero(ELTYPE))
                    set_particle_position(u, system, candidate, typemax(ELTYPE))
                end
            end
        end
    end

    return system
end
