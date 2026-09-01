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

    # Set spatial min spacing (Alg. 4)
    foreach_system(semi) do neighbor_system
        spatial_min_spacing!(fluid_system, neighbor_system, v_ode, u_ode, semi)
    end

    # Set solution min spacing (Alg. 4)
    for criterion in criteria
        apply_criterion!(criterion, fluid_system, v_ode, u_ode, semi)
        # solution_min_spacing!(criterion, fluid_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline apply_criterion!(criterion::RefinementCriteria, system, v_ode, u_ode, semi) = system

# @inline function apply_criterion!(criterion::SolutionRefinementCriterion, system, v_ode, u_ode, semi)
#     (; satisfies_criterion) = criterion 

#     satisfies_criterion .= false 
#     criterion_condition!(criterion, system, v_ode, u_ode, semi)

#     return system 
# end 

@inline solution_min_spacing!(criterion::RefinementCriteria, system::AbstractFluidSystem,
                              v_ode, u_ode, semi) = system

@inline function solution_min_spacing!(criterion::SolutionRefinementCriterion,
                                       system::AbstractFluidSystem, v_ode, u_ode, semi)
    # TODO: If a fluid particles satisfies a the criteria for solution adaptivity, 
    # we set its spacing to the min spacing.
    # (; satisfies_criterion) = criterion
    # (; min_spacing) = fluid_system.particle_refinement
    # (; _particle_spacing) = fluid_system.cache

    # for particle in eachparticle(system)
    #     if satisfies_criterion[particle]
    #         _particle_spacing[particle] = min_spacing
    #     end 
    # end 

    return system
end

@inline spatial_min_spacing!(fluid_system::AbstractFluidSystem, neighbor_system, v_ode,
                             u_ode, semi) = fluid_system

# @inline function spatial_min_spacing!(fluid_system::AbstractFluidSystem, neighbor_system::Union{AbstractBoundarySystem, AbstractStructureSystem}, v_ode, u_ode, semi)
#     (; _particle_spacing) = fluid_system.cache
#     (; min_spacing) = fluid_system.particle_refinement

#     fluid_u = wrap_u(u_ode, fluid_system, semi)
#     fluid_coords = current_coordinates(fluid_u, fluid_system)
#     neighbor_u = wrap_u(u_ode, neighbor_system, semi)
#     neighbor_coords = current_coordinates(neighbor_u, neighbor_system)

#     foreach_point_neighbor(neighbor_system, fluid_system, neighbor_coords, fluid_coords, semi) do neighbor_particle, fluid_particle, pos_diff, distance
#         distance < sqrt(eps()) && return
#         _particle_spacing[fluid_particle] = min_spacing
#     end

#     return fluid_system
# end 

# For testing "taylor_greene_vortex_2d_refinement.jl"
# Mark a 0.4 x 0.4 box perfectly in the center for refinement. 
# struct BoxDebugRefinement{} <: SolutionRefinementCriterion 
# satisfies_criterion :: SC
# end

# TODO
# function BoxDebugRefinement()
# end

function apply_criterion!(criterion::SolutionRefinementCriterion, system, v_ode, u_ode,
                          semi)
    (; _particle_spacing) = system.cache
    (; min_spacing) = system.particle_refinement

    # Retrieve the unrefined background spacing from the initial condition
    background_spacing = system.initial_condition.particle_spacing

    u = wrap_u(u_ode, system, semi)
    for particle in eachparticle(system)
        pos = current_coords(u, system, particle)

        # Only split particles inside the centered box
        if (0.3 < pos[1] < 1.7) && (0.2 < pos[2] < 0.8)
            _particle_spacing[particle] = min_spacing
        else
            # If outside the box, force the target back to the coarse resolution!
            _particle_spacing[particle] = background_spacing
        end
    end

    return system
end
