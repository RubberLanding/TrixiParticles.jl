struct ResizeBuffer{IArray, FArray}
    n_add_particles       :: IArray
    n_delete_particles    :: IArray
    n_new_particles       :: IArray
    masses_new            :: FArray
    densities_new         :: FArray
    smoothing_lengths_new :: FArray
    velocities_new        :: FArray
    positions_new         :: FArray
end

function ResizeBuffer(initial_condition)
    # 1-element arrays for mutable counters
    n_add_particles = [0]
    n_delete_particles = [0]
    n_new_particles = [length(initial_condition.mass)]

    # 1D Vectors for physical properties
    masses_new = similar(initial_condition.mass, 0)
    densities_new = similar(initial_condition.density, 0)

    # Assuming smoothing length uses the same float type as mass
    smoothing_lengths_new = similar(initial_condition.mass, 0)

    # Flat 1D Vectors for multidimensional properties!
    velocities_new = similar(initial_condition.velocity, 0)
    positions_new = similar(initial_condition.coordinates, 0)

    return ResizeBuffer(n_add_particles, n_delete_particles, n_new_particles,
                        masses_new, densities_new,
                        smoothing_lengths_new, velocities_new, positions_new)
end

function Base.resize!(v_ode, u_ode, _v_ode, _u_ode, semi::Semidiscretization)
    # In case the total number of particles in the system decreased, 
    # swap the particles to be deleted to the end and resize the system. 
    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return

        refinement = system.particle_refinement
        (; n_new_particles, n_add_particles, n_delete_particles) = refinement.resize_buffer
        (n_add_particles[1] >= n_delete_particles[1]) && return

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        overwrite_swap(system, v, u, n_new_particles[1])
    end

    # Resize the system buffers
    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return

        n_new = system.particle_refinement.resize_buffer.n_new_particles[1]
        resize!(system, n_new)
    end

    # Resize v_ode, u_ode and the ranges of the Semidiscretization
    resize!(semi, v_ode, u_ode, _v_ode, _u_ode)

    # In case the total number of particles in the system increases,
    # append the new particles to the end. 
    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return

        (; n_add_particles, n_delete_particles) = system.particle_refinement.resize_buffer
        (n_add_particles[1] < n_delete_particles[1]) && return

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        overwrite_append(system, v, u)
    end

    # Update the current number of particles in the ResizeBuffer
    foreach_system(semi) do system
        !isa(system, AbstractFluidSystem) && return
        isnothing(system.particle_refinement) && return

        n_new = system.particle_refinement.resize_buffer.n_new_particles[1]
        fill!(system.particle_refinement.n_current_particles, n_new)
    end

    # Resize the neighborhood searches in the Semidiscretization
    new_semi = resize_semi(semi)
    reinitialize_neighborhood_searches!(new_semi, u_ode)

    return new_semi
end

# IMPORTANT: `indices_delete_particles` needs to be sorted
function overwrite_swap(system, v, u, n_new_particles)
    (; delete_candidates, n_current_particles, resize_buffer) = system.particle_refinement
    (; n_add_particles, masses_new, densities_new, smoothing_lengths_new, velocities_new,
     positions_new) = resize_buffer

    NDIMS = ndims(system)
    vel_view = reshape(velocities_new, NDIMS, :)
    pos_view = reshape(positions_new, NDIMS, :)

    # TODO: Implement without `findall()`
    # Automatically sorted
    indices_delete_particles = findall(delete_candidates)

    # Overwrite the particles to delete with new particles 
    for idx in 1:n_add_particles[1]
        idx_delete = indices_delete_particles[idx]

        set_particle_smoothing_length!(system, idx_delete, smoothing_lengths_new[idx])
        set_particle_mass!(system, idx_delete, masses_new[idx])
        set_particle_density!(v, system, idx_delete, densities_new[idx])
        @views set_particle_velocity!(v, system, idx_delete, vel_view[:, idx])
        @views set_particle_position!(u, system, idx_delete, pos_view[:, idx])
    end

    # Move the particles to be deleted to the end by swapping
    indices_delete_particles_ = indices_delete_particles[(n_add_particles[1] + 1):end]
    idx_swap = n_current_particles[1]

    for idx_delete in indices_delete_particles_
        # Nothing to do
        if idx_delete > n_new_particles
            continue
        end

        # Find the last index that is not to be deleted
        while delete_candidates[idx_swap] && idx_swap > (n_new_particles + 1)
            idx_swap -= 1
        end

        # Swap the particles at postion idx_swap and idx_delete
        # Get values of particle at position `idx_swap`
        smoothing_length_swap = smoothing_length(system, idx_swap)
        mass_swap = hydrodynamic_mass(system, idx_swap)
        density_swap = current_density(v, system, idx_swap)
        vel_swap = current_velocity(v, system, idx_swap)
        pos_swap = current_coords(u, system, idx_swap)

        # Copy particle to idx_delete
        set_particle_smoothing_length!(system, idx_delete, smoothing_length_swap)
        set_particle_mass!(system, idx_delete, mass_swap)
        set_particle_density!(v, system, idx_delete, density_swap)
        set_particle_velocity!(v, system, idx_delete, vel_swap)
        set_particle_position!(u, system, idx_delete, pos_swap)

        idx_swap -= 1
    end
end

function overwrite_append(system, v, u)
    (; delete_candidates, n_current_particles, resize_buffer) = system.particle_refinement
    (; n_add_particles, n_delete_particles,
     masses_new, densities_new, smoothing_lengths_new, velocities_new,
     positions_new) = resize_buffer

    NDIMS = ndims(system)
    vel_view = reshape(velocities_new, NDIMS, :)
    pos_view = reshape(positions_new, NDIMS, :)

    # TODO: Implement without `findall()`
    indices_delete_particles = findall(delete_candidates)

    # Overwrite the particles to delete with new particles 
    for i in 1:n_delete_particles[1]
        idx_delete = indices_delete_particles[i]

        set_particle_smoothing_length!(system, idx_delete, smoothing_lengths_new[i])
        set_particle_mass!(system, idx_delete, masses_new[i])
        set_particle_density!(v, system, idx_delete, densities_new[i])
        @views set_particle_velocity!(v, system, idx_delete, vel_view[:, i])
        @views set_particle_position!(u, system, idx_delete, pos_view[:, i])
    end

    n_particles_diff = n_add_particles[1] - n_delete_particles[1]
    # Append the rest of the new particles at the end
    for k in 1:n_particles_diff
        idx = n_current_particles[1] + k
        idx_new = n_delete_particles[1] + k

        set_particle_smoothing_length!(system, idx, smoothing_lengths_new[idx_new])
        set_particle_mass!(system, idx, masses_new[idx_new])
        set_particle_density!(v, system, idx, densities_new[idx_new])
        @views set_particle_velocity!(v, system, idx, vel_view[:, idx_new])
        @views set_particle_position!(u, system, idx, pos_view[:, idx_new])
    end
end

function Base.resize!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    (; systems) = semi

    # Backup current state to buffers and
    # resize buffers to match current length before copying
    resize!(_v_ode, length(v_ode))
    resize!(_u_ode, length(u_ode))
    copyto!(_v_ode, v_ode)
    copyto!(_u_ode, u_ode)

    # Calculate new ranges and sizes
    sizes_u_new = [u_nvariables(system) * nparticles_new(system)
                   for system in systems]
    sizes_v_new = [v_nvariables(system) * nparticles_new(system)
                   for system in systems]

    ranges_u_new = [(sum(sizes_u_new[1:(i - 1)]) + 1):sum(sizes_u_new[1:i])
                    for i in eachindex(sizes_u_new)]
    ranges_v_new = [(sum(sizes_v_new[1:(i - 1)]) + 1):sum(sizes_v_new[1:i])
                    for i in eachindex(sizes_v_new)]

    size_v_new, size_u_new = sum(sizes_v_new), sum(sizes_u_new)

    # Resize u and v to the new size 
    resize!(v_ode, size_v_new)
    resize!(u_ode, size_u_new)

    # Copy from the buffer
    for i in eachindex(systems)
        r_u_old = semi.ranges_u[i]
        r_v_old = semi.ranges_v[i]

        r_u_new = ranges_u_new[i]
        r_v_new = ranges_v_new[i]

        # Calculate number of elements to copy
        len_u = min(length(r_u_old), length(r_u_new))
        len_v = min(length(r_v_old), length(r_v_new))

        # Syntax: copyto!(dest, dest_offset, src, src_offset, amount)
        if len_u > 0
            copyto!(u_ode, first(r_u_new), _u_ode, first(r_u_old), len_u)
        end
        if len_v > 0
            copyto!(v_ode, first(r_v_new), _v_ode, first(r_v_old), len_v)
        end

        # Update ranges
        semi.ranges_u[i] = r_u_new
        semi.ranges_v[i] = r_v_new
    end

    resize!(_v_ode, length(v_ode))
    resize!(_u_ode, length(u_ode))

    return v_ode
end

# TODO 
@inline Base.resize!(system, n) = system

function Base.resize!(system::AbstractFluidSystem, n)
    return resize!(system, system.particle_refinement, n)
end

@inline Base.resize!(system, ::Nothing, n) = system

function Base.resize!(system::WeaklyCompressibleSPHSystem, refinement, n)
    (; mass, pressure, smoothing_length) = system

    # Resize standard system properties
    resize!(mass, n)
    resize!(pressure, n)
    resize!(smoothing_length, n)

    # Resize the Density
    resize_density!(system, n, system.density_calculator)

    # Resize the Cache and Refinement tracking arrays
    resize_cache!(system, n)
    resize_refinement!(refinement, n)

    return system
end

# Should be called in `split.jl` and `merge.jl` directly until buffer approach is implemented
function resize_buffer!(buffer::ResizeBuffer, system::AbstractFluidSystem, n)
    (; masses_new, densities_new,
     smoothing_lengths_new, velocities_new, positions_new) = buffer
    NDIMS = ndims(system)

    resize!(masses_new, n)
    resize!(densities_new, n)
    resize!(smoothing_lengths_new, n)
    resize!(velocities_new, NDIMS * n)
    resize!(positions_new, NDIMS * n)
end

# TODO
function resize_refinement!(refinement::ParticleRefinement, n)
    (; delete_candidates, split_candidates, merge_candidates) = refinement

    resize!(delete_candidates, n)
    resize!(split_candidates, n)
    resize!(merge_candidates, n)

    return refinement
end

# TODO
# We need one of these for each type of cache (and for each type of `_FluidSystem`?)
function resize_cache!(system::WeaklyCompressibleSPHSystem, n)
    (; reference_mass, _particle_spacing, is_anchor_particle, candidate_flags,
     candidate_offsets, neighbor_mass, neighbor_count) = system.cache

    resize!(reference_mass, n)
    resize!(_particle_spacing, n)
    resize!(is_anchor_particle, n)
    resize!(candidate_flags, n)
    resize!(candidate_offsets, n)
    resize!(neighbor_mass, n)
    resize!(neighbor_count, n)

    return system
end

resize_density!(system, n, ::SummationDensity) = resize!(system.cache.density, n)
resize_density!(system, n, ::ContinuityDensity) = system

function reset_resize_buffer!(resize_buffer::ResizeBuffer, system::AbstractFluidSystem)
    (; n_new_particles, n_add_particles, n_delete_particles,
     masses_new, densities_new, smoothing_lengths_new, velocities_new,
     positions_new) = resize_buffer

    n_particles = nparticles(system)

    fill!(n_new_particles, n_particles)
    fill!(n_add_particles, 0)
    fill!(n_delete_particles, 0)

    fill!(masses_new, 0.0)
    fill!(densities_new, 0.0)
    fill!(smoothing_lengths_new, 0.0)
    fill!(velocities_new, 0.0)
    fill!(positions_new, 0.0)

    return resize_buffer
end

function resize_semi(semi::Semidiscretization)
    (; systems) = semi

    new_searches = map(systems) do system
        map(systems) do neighbor_system

            # Bypass `get_neighborhood_search` for TLSPH.
            # We do not refine TLSPH and do not want to modify its `self_interaction_nhs`.
            # Instead we take the nhs stored in the semidiscretization and resize it. 
            system_index = system_indices(system, semi)
            neighbor_index = system_indices(neighbor_system, semi)

            old_nhs = semi.neighborhood_searches[system_index][neighbor_index]
            n_new = nparticles(neighbor_system)
            copy_neighborhood_search(old_nhs, old_nhs.search_radius, n_new)
        end
    end

    return Semidiscretization(systems, semi.ranges_u, semi.ranges_v, new_searches,
                              semi.parallelization_backend, semi.update_callback_used,
                              semi.integrate_tlsph)
end
function reinitialize_neighborhood_searches!(semi, u_ode)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            reinitialize_neighborhood_search!(semi, system, neighbor, u_ode)
        end
    end
    return semi
end

function reinitialize_neighborhood_search!(semi, system, neighbor, u_ode)
    u_system = wrap_u(u_ode, system, semi)
    u_neighbor = wrap_u(u_ode, neighbor, semi)

    # TODO Initialize after adapting to the GPU.
    # Currently, this cannot use `semi.parallelization_backend`
    # because data is still on the CPU.   
    PointNeighbors.initialize!(get_neighborhood_search(system, neighbor, semi),
                               current_coords(u_system, system),
                               current_coords(u_neighbor, neighbor),
                               eachindex_y=each_active_particle(neighbor),
                               parallelization_backend=PolyesterBackend())

    return semi
end

function reinitialize_neighborhood_search!(semi, system::TotalLagrangianSPHSystem,
                                           neighbor::TotalLagrangianSPHSystem, u_ode)
    # For TLSPH, the self-interaction NHS is already initialized in the system constructor
    return semi
end
