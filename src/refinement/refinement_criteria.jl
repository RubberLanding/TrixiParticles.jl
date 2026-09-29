abstract type RefinementCriteria end

struct SpatialRefinementCriterion <: RefinementCriteria end

struct SolutionRefinementCriterion{F} <: RefinementCriteria
    condition_func :: F
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
    (; criteria) = fluid_system.particle_refinement

    # Initialize the target particle spacing buffer
    for particle in eachparticle(fluid_system)
        _particle_spacing[particle] = particle_spacing(fluid_system, particle)
    end

    # Set min spacing (Alg. 4)
    for criterion in criteria
        apply_criterion!(criterion, fluid_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline function apply_criterion!(criterion::SpatialRefinementCriterion, fluid_system, v_ode, u_ode, semi)
    # Set spatial min spacing (Alg. 4)
    foreach_system(semi) do neighbor_system
        spatial_min_spacing!(fluid_system, neighbor_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline spatial_min_spacing!(fluid_system::AbstractFluidSystem, neighbor_system, v_ode,
                             u_ode, semi) = fluid_system

@inline function spatial_min_spacing!(fluid_system::AbstractFluidSystem, neighbor_system::Union{AbstractBoundarySystem, AbstractStructureSystem}, v_ode, u_ode, semi)
    (; _particle_spacing) = fluid_system.cache
    (; min_spacing) = fluid_system.particle_refinement

    fluid_u = wrap_u(u_ode, fluid_system, semi)
    fluid_coords = current_coordinates(fluid_u, fluid_system)
    neighbor_u = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(neighbor_u, neighbor_system)

    foreach_point_neighbor(neighbor_system, fluid_system, neighbor_coords, fluid_coords, semi) do neighbor_particle, fluid_particle, pos_diff, distance
        distance < sqrt(eps()) && return
        _particle_spacing[fluid_particle] = min_spacing
    end

    return fluid_system
end 

# For testing "taylor_greene_vortex_2d_refinement.jl"
# Mark a 0.4 x 0.4 box perfectly in the center for refinement. 
# struct BoxDebugRefinement{} <: SolutionRefinementCriterion 
# satisfies_criterion :: SC
# end

# TODO
# function BoxDebugRefinement()
# end

@inline function apply_criterion!(criterion::SolutionRefinementCriterion, system, 
                                  v_ode, u_ode, semi)
    (; _particle_spacing) = system.cache
    (; min_spacing) = system.particle_refinement

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    for particle in eachparticle(system)
        if criterion.condition_func(system, particle, v, u, semi)
            _particle_spacing[particle] = min_spacing
        end
    end

    return system
end