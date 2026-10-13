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

@inline Base.resize!(system::AbstractSystem, n) = system

function Base.resize!(system::AbstractFluidSystem, n)
    return resize!(system, system.particle_refinement, n)
end

@inline Base.resize!(system, ::Nothing, n) = system

function Base.resize!(system::AbstractFluidSystem, refinement, n)
    # TrixiParticles.@autoinfiltrate
    # Resize standard system properties
    resize_system!(system, n)

    # Resize the density
    resize_density!(system, n, system.density_calculator)

    # Resize the cache and refinement arrays
    resize_cache!(system, n)

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
     grad_pressure, grad_velocity,
     delete_candidates, split_candidates, merge_candidates) = system.cache

    NDIMS = ndims(system)

    resize!(reference_mass, n)
    resize!(_particle_spacing, n)
    resize!(candidate_flags, n)
    resize!(candidate_offsets, n)
    resize!(neighbor_mass, n)
    resize!(neighbor_count, n)

    resize!(delete_candidates, n)
    resize!(split_candidates, n)
    resize!(merge_candidates, n)

    resize!(delta_v, n * NDIMS)
    resize!(grad_density, n * NDIMS)
    resize!(grad_pressure, n * NDIMS)
    resize!(grad_velocity, n * NDIMS * NDIMS)

    return system
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

# Quick fix for setting `eachpoint` for the TrivialNHS with Taylor Greene
function update_nhs(semi, system, neighbor)
    (; neighborhood_searches) = semi

    system_index = TrixiParticles.system_indices(system, semi)
    neighbor_index = TrixiParticles.system_indices(neighbor, semi)

    nhs = neighborhood_searches[system_index, neighbor_index]
    n_active_particles = system.buffer.active_particle_count[]
    neighborhood_searches[system_index,
                          neighbor_index] = TrixiParticles.copy_neighborhood_search(nhs,
                                                                                    nhs.search_radius,
                                                                                    n_active_particles)
end

function resize_integrator_cache!(integrator, semi, old_ranges_v, old_ranges_u)
    # Collect all history and intermediate caches
    caches_to_clean = [integrator.uprev, OrdinaryDiffEqCore.full_cache(integrator)...,
        integrator.fsalfirst, integrator.fsallast]

    # 1. SHIFT UPREV AND WIPE INTERMEDIATE CACHES
    for c in caches_to_clean
        # Skip invalid caches or caches of a different type
        if isnothing(c) || typeof(c) != typeof(integrator.u)
            continue
        end

        is_u = (c.x[1] === integrator.u.x[1])
        is_uprev = (c.x[1] === integrator.uprev.x[1])

        if is_uprev
            # `uprev` was resized but not shifted. We shift it right-to-left 
            # to safely push data into newly allocated space without overwriting.
            for sys_idx in reverse(eachindex(semi.systems))
                r_v_old = old_ranges_v[sys_idx]
                r_v_new = semi.ranges_v[sys_idx]

                r_u_old = old_ranges_u[sys_idx]
                r_u_new = semi.ranges_u[sys_idx]

                len_v = min(length(r_v_old), length(r_v_new))
                len_u = min(length(r_u_old), length(r_u_new))

                # Base.copyto! safely handles overlapping memory
                if len_v > 0
                    copyto!(c.x[1], first(r_v_new), c.x[1], first(r_v_old), len_v)
                end
                if len_u > 0
                    copyto!(c.x[2], first(r_u_new), c.x[2], first(r_u_old), len_u)
                end
            end
        elseif !is_u
            # INTERMEDIATE CACHES (k): Wipe completely to prevent boundary teleportation
            fill!(c.x[1], 0.0)
            fill!(c.x[2], 0.0)
        end
    end

    # 2. SANITIZE THE INACTIVE GAPS IN STATE ARRAYS
    for (sys_idx, sys) in enumerate(semi.systems)
        r_v_new = semi.ranges_v[sys_idx]
        r_u_new = semi.ranges_u[sys_idx]

        n_vars_v = size(TrixiParticles.wrap_v(integrator.u.x[1], sys, semi), 1)
        n_vars_u = size(TrixiParticles.wrap_u(integrator.u.x[2], sys, semi), 1)

        active_v_len = length(r_v_new)
        active_u_len = length(r_u_new)

        # Calculate true active lengths using the buffer[cite: 7]
        if hasproperty(sys, :buffer) && hasproperty(sys.buffer, :active_particle_count)
            active_v_len = sys.buffer.active_particle_count[] * n_vars_v
            active_u_len = sys.buffer.active_particle_count[] * n_vars_u
        end

        inactive_v_range = (first(r_v_new) + active_v_len):last(r_v_new)
        inactive_u_range = (first(r_u_new) + active_u_len):last(r_u_new)

        # Zero out the inactive gaps in both `u` and `uprev`
        for c in caches_to_clean
            if !isnothing(c) && typeof(c) == typeof(integrator.u)
                if c.x[1] === integrator.u.x[1] || c.x[1] === integrator.uprev.x[1]
                    if length(inactive_v_range) > 0
                        c.x[1][inactive_v_range] .= 0.0
                    end
                    if length(inactive_u_range) > 0
                        c.x[2][inactive_u_range] .= 0.0
                    end
                end
            end
        end
    end
end
