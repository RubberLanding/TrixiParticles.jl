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
# Update the reference spacing of all fluid particles
@inline function update_particle_spacing(fluid_system::AbstractFluidSystem, refinement,
                                         v_ode, u_ode, semi)
    (; spacing_ratio, smoothing_length_factor) = refinement
    (; _particle_spacing, reference_mass) = fluid_system.cache

    NDIMS = ndims(fluid_system)
    spacing_ratio_3 = spacing_ratio^3

    u = wrap_u(u_ode, fluid_system, semi)
    v = wrap_v(v_ode, fluid_system, semi)
    system_coords = current_coordinates(u, fluid_system)

    for particle in eachparticle(fluid_system)
        spacing_min, spacing_max,
        spacing_avg = min_max_avg_spacing(fluid_system, semi, u_ode, system_coords,
                                          particle)
        # TrixiParticles.@autoinfiltrate
        # TODO: Check for `spacing_min == 0.0` to avoid division-by-zero.
        if spacing_max / spacing_min < spacing_ratio_3
            new_spacing = min(spacing_max, spacing_ratio * spacing_min)
            # TrixiParticles.@autoinfiltrate
        else
            new_spacing = spacing_avg
            # TrixiParticles.@autoinfiltrate
        end

        reference_mass[particle] = current_density(v, fluid_system, particle) *
                                   new_spacing^NDIMS
    end

    # TrixiParticles.@autoinfiltrate

    return fluid_system
end

# Compute the minimum, maximum, and average reference spacing in the neighborhood around a particle
@inline function min_max_avg_spacing(system, semi, u_ode, system_coords, particle)
    (; _particle_spacing) = system.cache

    spacing_min = Inf
    spacing_max = zero(eltype(system))
    spacing_avg = zero(eltype(system))
    counter_neighbors = 0

    foreach_system(semi) do neighbor_system
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)

        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        PointNeighbors.foreach_neighbor(system_coords, neighbor_coords, neighborhood_search,
                                        particle) do particle, neighbor, pos_diff, distance
            neighbor_spacing = _particle_spacing[neighbor]
            spacing_min = min(spacing_min, neighbor_spacing)
            spacing_max = max(spacing_max, neighbor_spacing)
            spacing_avg += neighbor_spacing
            counter_neighbors += 1
        end
    end

    if counter_neighbors == 0
        spacing_particle = particle_spacing(system, particle)
        return spacing_particle, spacing_particle, spacing_particle
    else
        spacing_avg = spacing_avg / counter_neighbors
    end

    return spacing_min, spacing_max, spacing_avg
end
