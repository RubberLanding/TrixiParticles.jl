# TODO: Make sure that the following functions are only called from a callback
# First we resize the actual system and all the objects containing the properties like mass, pressure and so on.
# Then we update the semidiscretization, u and v to reflect this change. 
function Base.resize!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    (; systems) = semi
    # Resize all systems
    foreach_system(semi) do system
        resize!(system, capacity(system))
    end

    copyto!(_v_ode, v_ode)
    copyto!(_u_ode, u_ode)

    # Get ranges after resizing the systems
    ranges_v_new, ranges_u_new = ranges_vu(systems)
    ranges_v_old = copy(semi.ranges_v)
    ranges_u_old = copy(semi.ranges_u)

    # Update ranges after resizing the systems
    for i in eachindex(systems)
        semi.ranges_v[i] = ranges_v_new[i]
        semi.ranges_u[i] = ranges_u_new[i]
    end

    # Accumulate all sizes into the total size of u and v for semi
    size_u_new = sum(u_nvariables(system) * n_integrated_particles(system)
                     for system in systems)
    size_v_new = sum(v_nvariables(system) * n_integrated_particles(system)
                     for system in systems)

    # After `deleteat!`, there are fewer particles, and the `deleteat!(system, v, u)` function
    # ensures that all rejected values are stored at the tail of `v_ode` and `u_ode`.
    if length(v_ode) < size_v_new
        # Resize before copy
        resize!(v_ode, size_v_new)
        resize!(_v_ode, size_u_new)

        resize!(u_ode, size_v_new)
        resize!(_u_ode, size_u_new)
    end

    # Copy the values of the old ranges to the new ranges. 
    # Only consider overlapping values that exist in both ranges.
    for i in eachindex(ranges_u_old)
        length_u = min(length(ranges_u_old[i]), length(ranges_u_new[i]))
        for j in 0:(length_u - 1)
            u_ode[ranges_u_new[i][1] + j] = _u_ode[ranges_u_old[i][1] + j]
        end

        length_v = min(length(ranges_v_old[i]), length(ranges_v_new[i]))
        for j in 0:(length_v - 1)
            v_ode[ranges_v_new[i][1] + j] = _v_ode[ranges_v_old[i][1] + j]
        end
    end

    if length(v_ode) > size_v_new
        # Resize after copy
        resize!(v_ode, size_v_new)
        resize!(u_ode, size_u_new)

        resize!(_v_ode, size_v_new)
        resize!(_u_ode, size_u_new)
    end

    return v_ode
end

Base.resize!(system::AbstractSystem, capacity_system) = system

function Base.deleteat!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    # Delete at specific indices
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        deleteat!(system, v, u)
    end

    resize!(semi, v_ode, u_ode, _v_ode, _u_ode)

    return semi
end

Base.deleteat!(system::AbstractSystem, v, u) = system

# Swap a particle to delete with one at the tail. 
# When `resize!` is called later, it will crop of all particles at the tail, deleting them.
function Base.deleteat!(system::AbstractFluidSystem, v, u)
    (; cache) = system

    isempty(cache.delete_candidates) && return

    delete_counter = 0
    for particle in eachparticle(system)
        if cache.delete_candidates[particle]
            dump_id = nparticles(system) - delete_counter
            # Swap mass
            system.mass[particle] = hydrodynamic_mass(system, dump_id)

            # TODO: Swap smoothing lengths
            # smoothing_length_keep = smoothing_length(system, dump_id)
            # set_particle_smoothing_length!(system, particle, smoothing_length_keep)

            # Swap pressure
            pressure_keep = current_pressure(v, system, dump_id)
            set_particle_pressure!(v, system, particle, pressure_keep)
            # Swap density
            density_keep = current_density(v, system, dump_id)
            set_particle_density!(v, system, particle, density_keep)
            # Swap velocity and position
            vel_keep = current_velocity(v, system, dump_id)
            pos_keep = current_coords(u, system, dump_id)
            for dim in 1:ndims(system)
                v[dim, particle] = vel_keep[dim]
                u[dim, particle] = pos_keep[dim]
            end
            delete_counter += 1
        end
    end

    cache.additional_capacity[] -= delete_counter
    resize!(cache.delete_candidates, 0)
    cache.values_conserved[] = true

    return system
end

@inline capacity(system) = nparticles(system)

@inline function capacity(system::AbstractFluidSystem)
    (; cache) = system
    return nparticles(system) + cache.additional_capacity[]
end
