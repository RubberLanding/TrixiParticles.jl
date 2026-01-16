abstract type RefinementCriteria end

struct SpatialRefinementCriterion <: RefinementCriteria end

struct SolutionRefinementCriterion <: RefinementCriteria end

struct DummyRefinementCriterion <: RefinementCriteria end

function apply_refinement_criteria!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        apply_refinement_criteria!(system, v_ode, u_ode, semi)
    end
end

@inline apply_refinement_criteria!(system, v_ode, u_ode, semi) = system

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    apply_refinement_criteria!(system, system.particle_refinement, v, u, v_ode, u_ode, semi)
end

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, ::Nothing,
                                            v, u, v_ode, u_ode, semi)
    system
end

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, refinement,
                                            v, u, v_ode, u_ode, semi)
    (; criteria) = system.particle_refinement

    for criterion in criteria
        criterion(system, v, u, v_ode, u_ode, semi)
    end
end

@inline function (criterion::SpatialRefinementCriterion)(system, v, u, v_ode, u_ode, semi)
    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        set_refinement_spacing!(system, neighbor_system, system_coords, v_ode, u_ode, semi)
    end
    return system
end

@inline function (criterion::DummyRefinementCriterion)(system, v, u, v_ode, u_ode, semi)
    return system
end

@inline set_refinement_spacing!(system, _, _, _, _, _) = system

@inline function set_refinement_spacing!(particle_system::AbstractFluidSystem,
                                       neighbor_system::Union{AbstractBoundarySystem,
                                                              AbstractStructureSystem},
                                       system_coords, v_ode, u_ode, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return
        
        # QUESTION: So we repeatedly overwrite the spacing for a particle? Do we want that?
        spacing_neighbor = TrixiParticles.particle_spacing(neighbor_system, neighbor)
        spacing_particle = TrixiParticles.particle_spacing(particle_system, particle)
        set_particle_spacing!(particle_system, particle,
                              min(spacing_neighbor, spacing_particle))
    end

    return particle_system
end
