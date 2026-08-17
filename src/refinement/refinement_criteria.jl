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
    apply_refinement_criteria!(system, system.particle_refinement, v_ode, u_ode, semi)
end

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode, semi)
    system
end

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, refinement, v_ode, u_ode, semi)
    (; criteria) = system.particle_refinement

    for criterion in criteria
        criterion(system, v_ode, u_ode, semi)
    end
end

@inline function (criterion::SpatialRefinementCriterion)(system, v_ode, u_ode, semi)
    u = wrap_u(u_ode, system, semi)
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
        
        spacing_neighbor = TrixiParticles.particle_spacing(neighbor_system, neighbor)
        spacing_particle = TrixiParticles.particle_spacing(particle_system, particle)

        # Implicitely store the new particle spacing by updating the particle's smoothing length 
        min_spacing = min(spacing_neighbor, spacing_particle)
        min_smoothing_length = particle_system.particle_refinement.smoothing_length_factor * min_spacing
        set_particle_smoothing_length!(particle_system, particle, min_smoothing_length)
    end

    return particle_system
end

# JUST FOR TESTING TAYLOR GREENE VORTEX WITH REFINEMENT 
@inline function (criterion::SolutionRefinementCriterion)(system, v_ode, u_ode, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    
    TrixiParticles.@threaded semi for particle in TrixiParticles.eachparticle(system)
        pos = TrixiParticles.current_coords(u, system, particle)
        
        # This safely defaults all particles to NOT be merge anchors!
        system.cache.reference_mass[particle] = TrixiParticles.hydrodynamic_mass(system, particle)
        system.cache.is_anchor_particle[particle] = false
        
        # Only split a tiny 0.4 x 0.4 box perfectly in the center.
        if (0.3 < pos[1] < 0.7) && (0.3 < pos[2] < 0.7)
            system.cache._particle_spacing[particle] = system.particle_refinement.min_spacing
            
            # This triggers splitting safely because target_mass < current_mass.
            target_mass = TrixiParticles.current_density(v, system, particle) * system.particle_refinement.min_spacing^2
            system.cache.reference_mass[particle] = target_mass
            
            # NO ANCHOR FLAG HERE! We want to split, not merge!
        end
    end
    return system
end