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
    (; is_fixed) = fluid_system.cache
    (; criteria) = refinement

    # The target spacing `_particle_spacing` is persistent and is not reset here.
    # Only reset which particles are fixed in this call.
    fill!(is_fixed, false)

    # Set the target spacing (Alg. 4). The spatial criteria always run first,
    # since they overwrite the spacing of the previous call for unfixed particles.
    # The solution criteria can then only decrease the spacing.
    for criterion in criteria
        apply_spatial_criterion!(criterion, fluid_system, v_ode, u_ode, semi)
    end

    for criterion in criteria
        apply_solution_criterion!(criterion, fluid_system, v_ode, u_ode, semi)
    end

    return fluid_system
end

@inline apply_spatial_criterion!(criterion, fluid_system, v_ode, u_ode, semi) = fluid_system

@inline function apply_spatial_criterion!(criterion::SpatialRefinementCriterion,
                                          fluid_system, v_ode, u_ode, semi)
    # Set the spacing of fluid particles near boundaries to the boundary spacing
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
    (; _particle_spacing, is_fixed) = fluid_system.cache
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
            # The first boundary in this call overwrites the spacing of the previous call,
            # further boundaries can only decrease it.
            if is_fixed[fluid_particle]
                _particle_spacing[fluid_particle] = min(_particle_spacing[fluid_particle],
                                                        neighbor_system_spacing)
            else
                _particle_spacing[fluid_particle] = neighbor_system_spacing
                is_fixed[fluid_particle] = true
            end
        end
    end

    return fluid_system
end

@inline apply_solution_criterion!(criterion, system, v_ode, u_ode, semi) = system

@inline function apply_solution_criterion!(criterion::SolutionRefinementCriterion, system,
                                           v_ode, u_ode, semi)
    (; _particle_spacing, is_fixed) = system.cache
    (; max_spacing) = system.particle_refinement
    (; get_target_spacing) = criterion

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    @threaded semi for particle in eachparticle(system)
        # The criterion is only active where it requests a spacing below `max_spacing`
        target_spacing = get_target_spacing(system, particle, v, u, semi)
        target_spacing < max_spacing || return

        # A solution criterion can only decrease the spacing. Near boundaries, this is
        # the boundary spacing set above, elsewhere the spacing of the previous call.
        _particle_spacing[particle] = min(_particle_spacing[particle], target_spacing)
        is_fixed[particle] = true
    end

    return system
end
