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
@inline function update_particle_spacing(system::AbstractFluidSystem, refinement,
                                         v_ode, u_ode, semi)
    (; spacing_ratio, smoothing_length_factor, min_spacing) = refinement
    (; _particle_spacing, reference_mass, is_anchor_particle) = system.cache

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    system_coords = current_coordinates(u, system)

    fill!(is_anchor_particle, false)

    foreach_system(semi) do solid_system
        # TODO: Extend this to other solid systems, e.g. moving systems like `TotalLagrangianSPHSystem` or `DEMSystem` 
        if solid_system isa AbstractBoundarySystem
            neighborhood_search = get_neighborhood_search(system, solid_system, semi)
            u_solid = wrap_u(u_ode, solid_system, semi)
            solid_coords = current_coordinates(u_solid, solid_system)

            foreach_point_neighbor(system, solid_system, system_coords, solid_coords, semi) do fluid_particle, solid_particle, pos_diff, distance
                _particle_spacing[fluid_particle] = min_spacing
                reference_mass[fluid_particle] = current_density(v, system, fluid_particle) * min_spacing^(ndims(system))
                is_anchor_particle[fluid_particle] = true
            end
        end
    end

    for particle in eachparticle(system)
        is_anchor_particle[particle] && continue

        spacing_min, spacing_max,
        spacing_avg = min_max_avg_spacing(system, semi, u_ode, system_coords,
                                          particle)

        if spacing_max / spacing_min < spacing_ratio^3
            new_spacing = min(spacing_max, spacing_ratio * spacing_min)
        else
            new_spacing = spacing_avg
        end

        _particle_spacing[particle] = new_spacing
        reference_mass[particle] = current_density(v, system, particle) *
                                   new_spacing^(ndims(system))
    end

    for particle in eachparticle(system)
        particle_smoothing_length = _particle_spacing[particle] * smoothing_length_factor
        set_particle_smoothing_length!(system, particle, particle_smoothing_length)
    end

    return system
end

# Compute the minimum, maximum, and average reference spacing in the neighborhood around a particle
@inline function min_max_avg_spacing(system, semi, u_ode, system_coords, particle)
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
            neighbor_spacing = particle_spacing(neighbor_system, neighbor)
            spacing_min = min(spacing_min, neighbor_spacing)
            spacing_max = max(spacing_max, neighbor_spacing)
            spacing_avg += neighbor_spacing
            counter_neighbors += 1
        end
    end

    if counter_neighbors != 0
        spacing_avg = spacing_avg / counter_neighbors
    end

    return spacing_min, spacing_max, spacing_avg
end