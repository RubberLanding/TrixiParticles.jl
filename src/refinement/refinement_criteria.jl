abstract type RefinementCriteria end

struct SpatialRefinementCriterion <: RefinementCriteria end

struct SolutionRefinementCriterion{F} <: RefinementCriteria
    get_target_spacing::F
end

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

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, ::Nothing, v_ode,
                                            u_ode, semi)
    system
end

@inline function apply_refinement_criteria!(fluid_system::AbstractFluidSystem, refinement,
                                            v_ode, u_ode, semi)
    (; _particle_spacing) = fluid_system.cache
    (; criteria, max_spacing) = fluid_system.particle_refinement

    # Initialize the target particle spacing buffer
    @threaded semi for particle in eachparticle(fluid_system)
        _particle_spacing[particle] = max_spacing
    end

    # Set min spacing (Alg. 4)
    for criterion in criteria
        apply_criterion!(criterion, fluid_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline function apply_criterion!(criterion::SpatialRefinementCriterion, fluid_system,
                                  v_ode, u_ode, semi)
    (; is_boundary_adjacent) = fluid_system.cache
    is_boundary_adjacent .= false

    # Set spatial min spacing (Alg. 4)
    foreach_system(semi) do neighbor_system
        spatial_min_spacing!(fluid_system, neighbor_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline spatial_min_spacing!(fluid_system::AbstractFluidSystem, neighbor_system, v_ode,
                             u_ode, semi) = fluid_system

@inline function spatial_min_spacing!(fluid_system::AbstractFluidSystem,
                                      neighbor_system::Union{OpenBoundarySystem,
                                                             WallBoundarySystem,
                                                             AbstractStructureSystem},
                                      v_ode, u_ode, semi)
    (; _particle_spacing, is_boundary_adjacent) = fluid_system.cache
    neighbor_system_spacing = neighbor_system.initial_condition.particle_spacing

    fluid_u = wrap_u(u_ode, fluid_system, semi)
    fluid_coords = current_coordinates(fluid_u, fluid_system)
    neighbor_u = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(neighbor_u, neighbor_system)
    neighborhood_search = get_neighborhood_search(fluid_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    @threaded semi for fluid_particle in eachparticle(fluid_system)
        @inbounds foreach_symmetric_neighbor(fluid_system, neighbor_system,
                                            fluid_coords, neighbor_coords,
                                            neighborhood_search, backend,
                                            fluid_particle) do fluid_particle, _, _, _
            is_boundary_adjacent[fluid_particle] = true
            _particle_spacing[fluid_particle] = min(_particle_spacing[fluid_particle],
                                                    neighbor_system_spacing)
        end
    end

    return fluid_system
end

@inline function apply_criterion!(criterion::SolutionRefinementCriterion, system,
                                  v_ode, u_ode, semi)
    (; _particle_spacing, is_boundary_adjacent) = system.cache
    (; get_target_spacing) = criterion

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    @threaded semi for particle in eachparticle(system)
        is_boundary_adjacent[particle] && return
        target_spacing = get_target_spacing(system, particle, v, u, semi)
        _particle_spacing[particle] = min(_particle_spacing[particle], target_spacing)
    end

    return system
end
