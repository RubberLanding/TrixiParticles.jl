function Base.resize!(v_ode, u_ode, _v_ode, _u_ode, semi::Semidiscretization)
    # This resizes the global v_ode, u_ode arrays and the ranges of the 
    # semidiscretization, according to the new size of the systems.
    resize!(semi, v_ode, u_ode, _v_ode, _u_ode)

    # Resize the neighborhood searches in the Semidiscretization
    resize_semi!(semi)
    reinitialize_neighborhood_searches!(semi, u_ode)

    return semi
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
    sizes_u_new = [u_nvariables(system) * n_integrated_particles(system)
                   for system in systems]
    sizes_v_new = [v_nvariables(system) * n_integrated_particles(system)
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

function Base.resize!(system::AbstractFluidSystem, refinement, n)
    # Resize standard system properties
    resize_system!(system, n)

    # Resize the density
    resize_density!(system, n, system.density_calculator)

    # Resize the cache and refinement arrays
    resize_cache!(system, n)
    resize_refinement!(refinement, n)

    # Resize the system buffer
    resize_system_buffer!(system.buffer, n)

    return system
end

@inline resize_system!(system, n) = system

@inline function resize_system!(system::EntropicallyDampedSPHSystem, n)
    (; mass, smoothing_length) = system

    resize!(mass, n)
    resize!(smoothing_length, n)

    return system
end

@inline function resize_system!(system::WeaklyCompressibleSPHSystem, n)
    (; mass, smoothing_length, pressure) = system

    resize!(mass, n)
    resize!(smoothing_length, n)
    resize!(pressure, n)

    return system
end

@inline function resize_density!(system, n, ::SummationDensity)
    resize!(system.cache.density, n)

    return system
end

@inline resize_density!(system, n, ::ContinuityDensity) = system

function resize_cache!(system::AbstractFluidSystem, n)
    resize_system_cache!(system, n)
    resize_refinement_cache!(system, n)

    return system
end

# TODO: Resize data specific to WCSPH cache
function resize_system_cache!(system::WeaklyCompressibleSPHSystem, n)
    return cache
end

# TODO: Resize data specific to EDAC cache
function resize_system_cache!(system::EntropicallyDampedSPHSystem, n)
    (; cache) = system
    (; pressure_average, neighbor_counter, beta) = cache

    resize!(pressure_average, n)
    resize!(neighbor_counter, n)
    resize!(beta, n)

    return cache
end

function resize_refinement_cache!(system::AbstractFluidSystem, n)
    (; reference_mass, _particle_spacing, candidate_flags,
     candidate_offsets, neighbor_mass, neighbor_count, delta_v, grad_density,
     grad_pressure, grad_velocity) = system.cache

    NDIMS = ndims(system)

    resize!(reference_mass, n)
    resize!(_particle_spacing, n)
    resize!(candidate_flags, n)
    resize!(candidate_offsets, n)
    resize!(neighbor_mass, n)
    resize!(neighbor_count, n)
    resize!(delta_v, n * NDIMS)
    resize!(grad_density, n * NDIMS)
    resize!(grad_pressure, n * NDIMS)
    resize!(grad_velocity, n * NDIMS * NDIMS)

    return system
end

# TODO
function resize_refinement!(refinement::ParticleRefinement, n)
    (; delete_candidates, split_candidates, merge_candidates) = refinement

    resize!(delete_candidates, n)
    resize!(split_candidates, n)
    resize!(merge_candidates, n)

    return refinement
end

@inline function resize_system_buffer!(buffer::SystemBuffer, n)
    (; active_particle, eachparticle) = buffer

    old_capacity = length(active_particle)
    n <= old_capacity && return buffer

    resize!(active_particle, n)
    active_particle[(old_capacity + 1):end] .= false

    resize!(eachparticle, n)
    
    return buffer
end

function resize_semi!(semi::Semidiscretization)
    (; systems, neighborhood_searches) = semi

    for system in systems
        for neighbor_system in systems

            # Bypass `get_neighborhood_search` for TLSPH.
            # We do not refine TLSPH and do not want to modify its `self_interaction_nhs`.
            # Instead we take the nhs stored in the semidiscretization and resize it. 
            system_index = system_indices(system, semi)
            neighbor_index = system_indices(neighbor_system, semi)

            old_nhs = neighborhood_searches[system_index, neighbor_index]
            n_new = nparticles(neighbor_system)
            neighborhood_searches[system_index,
                                  neighbor_index] = copy_neighborhood_search(old_nhs,
                                                                             old_nhs.search_radius,
                                                                             n_new)
        end
    end

    return semi
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
                               current_coordinates(u_system, system),
                               current_coordinates(u_neighbor, neighbor),
                               eachindex_y=each_active_particle(neighbor),
                               parallelization_backend=PolyesterBackend())

    return semi
end

function reinitialize_neighborhood_search!(semi, system::TotalLagrangianSPHSystem,
                                           neighbor::TotalLagrangianSPHSystem, u_ode)
    # For TLSPH, the self-interaction NHS is already initialized in the system constructor
    return semi
end