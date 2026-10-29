function update_particle_spacing(semi, v_ode, u_ode)
    foreach_system(semi) do system
        update_particle_spacing(system, v_ode, u_ode, semi)
    end
end

# The methods for the `AbstractFluidSystem` are defined in `src/schemes/fluid/fluid.jl`
@inline update_particle_spacing(system, v_ode, u_ode, semi) = system

@inline function update_particle_spacing(system::AbstractFluidSystem, v_ode, u_ode, semi)
    update_particle_spacing(system, system.particle_refinement, v_ode, u_ode, semi)
end

@inline update_particle_spacing(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode,
                                semi) = system

# Update the target spacing of all fluid particles that are not fixed by a criterion (Alg. 1).
# The target spacing is persistent, so the new spacing is computed from the spacing
# of the previous call. This way, the refinement band grows by one neighborhood per call.
@inline function update_particle_spacing(fluid_system::AbstractFluidSystem, refinement,
                                         v_ode, u_ode, semi)
    (; spacing_ratio, max_spacing, reference_density) = refinement
    (; _particle_spacing, _particle_spacing_tmp, reference_mass, is_fixed) = fluid_system.cache

    NDIMS = ndims(fluid_system)
    spacing_ratio_3 = spacing_ratio^3

    u = wrap_u(u_ode, fluid_system, semi)
    system_coords = current_coordinates(u, fluid_system)

    # Write to a temporary buffer, so that all particles use the spacing of the previous call
    @threaded semi for particle in eachparticle(fluid_system)
        if is_fixed[particle]
            _particle_spacing_tmp[particle] = _particle_spacing[particle]
        else
            spacing_min, spacing_max,
            spacing_avg = min_max_avg_spacing(fluid_system, semi, u_ode, system_coords,
                                              particle)

            # Note that `max_spacing` is the global maximum spacing and not the maximum
            # in the neighborhood. Otherwise, a uniform region could never coarsen.
            if spacing_max / spacing_min < spacing_ratio_3
                new_spacing = min(max_spacing, spacing_ratio * spacing_min)
            else
                new_spacing = spacing_avg
            end

            _particle_spacing_tmp[particle] = new_spacing
        end
    end

    @threaded semi for particle in eachparticle(fluid_system)
        new_spacing = _particle_spacing_tmp[particle]
        _particle_spacing[particle] = new_spacing
        reference_mass[particle] = reference_density * new_spacing^NDIMS
    end

    return fluid_system
end

# Compute the minimum, maximum, and geometric mean of the target spacing in the neighborhood
# around a particle. As in the reference code, only fluid neighbors are considered.
# Boundaries enter the spacing update through the fixed particles.
@inline function min_max_avg_spacing(system, semi, u_ode, system_coords, particle)
    ELTYPE = eltype(system)
    spacing_min = typemax(ELTYPE)
    spacing_max = zero(ELTYPE)
    log_spacing_sum = zero(ELTYPE)
    counter_neighbors = 0
    particle_radius = compact_support(system, system, particle)
    backend = semi.parallelization_backend

    foreach_system(semi) do neighbor_system
        neighbor_system isa AbstractFluidSystem || return

        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)

        @inbounds foreach_neighbor(system_coords, neighbor_coords,
                                   neighborhood_search, backend,
                                   particle;
                                   search_radius=particle_radius) do particle, neighbor,
                                                                     _, _
            neighbor_spacing = target_spacing(neighbor_system, neighbor)
            spacing_min = min(spacing_min, neighbor_spacing)
            spacing_max = max(spacing_max, neighbor_spacing)
            log_spacing_sum += log(neighbor_spacing)
            counter_neighbors += 1
        end
    end

    if counter_neighbors == 0
        spacing_particle = target_spacing(system, particle)
        return spacing_particle, spacing_particle, spacing_particle
    end

    # Geometric mean as in the reference code. Summing the logarithms instead of
    # multiplying the spacings avoids an underflow of the product for many neighbors.
    spacing_avg = exp(log_spacing_sum / counter_neighbors)

    return spacing_min, spacing_max, spacing_avg
end

# The persistent target spacing for systems with refinement, the actual spacing otherwise
@inline function target_spacing(system, particle)
    return target_spacing(system, get_refinement(system), particle)
end

@inline target_spacing(system, ::Nothing, particle) = particle_spacing(system, particle)

@inline function target_spacing(system, refinement, particle)
    return system.cache._particle_spacing[particle]
end
