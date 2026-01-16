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
    (; particle_spacing) = system
    (; max_spacing_ratio) = refinement
    (; _particle_spacing, reference_mass) = system.cache

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    system_coords = current_coordinates(u, system)

    for particle in eachparticle(system)
        spacing_min, spacing_max,
        spacing_avg = min_max_avg_spacing(system, semi, system_coords,
                                          particle)

        if spacing_max / spacing_min < max_spacing_ratio^3
            new_spacing = min(spacing_max, max_spacing_ratio * spacing_min)
        else
            new_spacing = spacing_avg
        end

        _particle_spacing[particle] = new_spacing
        reference_mass[particle] = current_density(v, system, particle) *
                                   new_spacing^(ndims(system))
    end

    particle_spacing .= _particle_spacing

    return system
end

# Compute the minimum, maximum, and average reference spacing in the neighborhood around a particle
@inline function min_max_avg_spacing(system, semi, u_ode, system_coords, particle)
    spacing_min = Inf
    spacing_max = zero(eltype(system))
    spacing_avg = zero(eltype(system))
    counter_neighbors = 0

    # QUESTION: Should `neighbor_system` be limited to AbstractFluidSystem? 
    # Why not just do a neighborhood search over all particles in `system` (see below)?
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

# Compute the minimum, maximum, and average reference spacing in the neighborhood around a particle
# This only look for the neighbors of a particle within system. 
@inline function min_max_avg_spacing(system, semi, system_coords, particle)
    spacing_min = Inf
    spacing_max = zero(eltype(system))
    spacing_avg = zero(eltype(system))
    counter_neighbors = 0

    neighborhood_search = get_neighborhood_search(system, system, semi)
    PointNeighbors.foreach_neighbor(system_coords, system_coords, neighborhood_search,
                                    particle) do particle, neighbor, pos_diff, distance
        neighbor_spacing = particle_spacing(system, neighbor)
        spacing_min = min(spacing_min, neighbor_spacing)
        spacing_max = max(spacing_max, neighbor_spacing)
        spacing_avg += neighbor_spacing
        counter_neighbors += 1
    end

    if counter_neighbors != 0
        spacing_avg = spacing_avg / counter_neighbors
    end

    return spacing_min, spacing_max, spacing_avg
end
