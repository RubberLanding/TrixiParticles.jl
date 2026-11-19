abstract type AbstractRefinementCriterion end

"""
    SpatialRefinementCriterion()

Refinement criterion that sets the target spacing of fluid particles near boundaries
to the particle spacing of the boundary.
A fluid particle is near a boundary if a boundary particle is within the compact support
of the fluid particle or of the boundary particle.
This applies to `WallBoundarySystem`s, structures and `OpenBoundarySystem`s.
"""
struct SpatialRefinementCriterion <: AbstractRefinementCriterion end

"""
    SolutionRefinementCriterion(target_spacing)

Refinement criterion that sets the target spacing of fluid particles based on the solution.
The function `target_spacing(system, particle, v, u, semi)` returns the requested target
spacing of `particle`. The criterion is only active where the requested spacing is smaller
than `max_spacing` of the [`ParticleRefinement`](@ref).

A solution criterion can only decrease the target spacing,
and it does not apply to particles near boundaries (see [`SpatialRefinementCriterion`](@ref)).
With multiple solution criteria, the smallest requested spacing is used.
"""
struct SolutionRefinementCriterion{F} <: AbstractRefinementCriterion
    target_spacing::F
end

@doc raw"""
    VorticityRefinementCriterion(; relative_threshold=0.05, absolute_threshold=0.01,
                                 target_spacing=nothing, outlet_cutoff=0.9,
                                 flow_direction=nothing)

Solution refinement criterion that refines fluid particles in regions of high vorticity
(solution adaptivity in Haftu et al.).

The vorticity ``\omega_a = \nabla \times v_a`` of each particle is computed as
```math
\omega_a = \sum_b \frac{m_b}{\rho_b} \nabla W_{ab} \times (v_b - v_a)
```
with the symmetric kernel gradient ``\nabla W_{ab}``.
The smoothed vorticity of a particle is the largest vorticity magnitude in its
neighborhood, ``w_a = \max_b |\omega_b|``, including the particle itself.
A particle is refined to `target_spacing` if
```math
w_a > \max(\text{relative\_threshold} \cdot \omega_{\max}, \text{absolute\_threshold}),
```
where ``\omega_{\max}`` is the largest vorticity magnitude of all particles of the system.

As a solution criterion, it can only decrease the target spacing,
and it does not apply to particles near boundaries (see [`SpatialRefinementCriterion`](@ref)).
The vorticity is only computed when the refinement runs.

# Keywords
- `relative_threshold=0.05`: Refine where the vorticity exceeds this fraction of the
                             largest vorticity in the system (5 % in Haftu et al.).
- `absolute_threshold=0.01`: Lower bound for the threshold, so that the criterion does not
                             refine a (nearly) resting fluid. The value has the unit of
                             the vorticity (1/s), so it depends on the flow.
- `target_spacing=nothing`: Target spacing of the refined particles.
                            The default is `min_spacing` of the [`ParticleRefinement`](@ref).
- `outlet_cutoff=0.9`: Only refine particles in the first fraction `outlet_cutoff` of the
                       initial fluid along `flow_direction`, that is, particles with
                       ``s < s_{\min} + \text{outlet\_cutoff} \, (s_{\max} - s_{\min})``,
                       where ``s = x \cdot d`` is the position along the flow direction
                       ``d`` and ``s_{\min}``, ``s_{\max}`` are the smallest and largest
                       ``s`` of the initial fluid. This avoids refining vortices that are
                       about to leave the domain. Pass `nothing` to refine everywhere.
- `flow_direction=nothing`: Direction of the flow for `outlet_cutoff`, e.g., `(1.0, 0.0)`.
                            The default is the positive x-direction.
                            The vector does not need to be normalized.
"""
struct VorticityRefinementCriterion{ELTYPE, TS, OC, FD} <: AbstractRefinementCriterion
    relative_threshold :: ELTYPE
    absolute_threshold :: ELTYPE
    target_spacing     :: TS     # `nothing` for `min_spacing`
    outlet_cutoff      :: OC     # `nothing` to refine everywhere
    flow_direction     :: FD     # `nothing` for the positive x-direction
end

function VorticityRefinementCriterion(; relative_threshold=0.05, absolute_threshold=0.01,
                                      target_spacing=nothing, outlet_cutoff=0.9,
                                      flow_direction=nothing)
    relative_threshold, absolute_threshold = promote(relative_threshold,
                                                     absolute_threshold)

    if !isnothing(flow_direction)
        if iszero(norm(flow_direction))
            throw(ArgumentError("`flow_direction` must not be zero"))
        end

        # Store an immutable copy, so that the criterion can be used on GPUs
        flow_direction = Tuple(flow_direction)
    end

    return VorticityRefinementCriterion(relative_threshold, absolute_threshold,
                                        target_spacing, outlet_cutoff, flow_direction)
end

# Set the target spacing of the particles in the regions of the refinement criteria
# and mark them as fixed for the target spacing update (Alg. 4 in Haftu et al.).
@inline apply_refinement_criteria!(system, v_ode, u_ode, semi) = system

@inline function apply_refinement_criteria!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return apply_refinement_criteria!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline apply_refinement_criteria!(system, ::Nothing, v_ode, u_ode, semi) = system

function apply_refinement_criteria!(system, refinement, v_ode, u_ode, semi)
    (; criteria) = refinement

    # The target spacing persists between refinement calls and is not reset here.
    # Only reset which particles are fixed in this call.
    set_zero!(system.cache.is_fixed)

    # The spatial criteria run first, since the solution criteria skip the particles
    # that the spatial criteria fixed (`MarkBoundary` before `SolutionAdaptive`
    # in the code of Haftu et al.).
    if any(criterion -> criterion isa SpatialRefinementCriterion, criteria)
        foreach_system(semi) do neighbor_system
            apply_spatial_criterion!(system, neighbor_system, v_ode, u_ode, semi)
        end
    end

    apply_solution_criteria!(system, criteria, v_ode, u_ode, semi)

    return system
end

@inline apply_spatial_criterion!(system, neighbor_system, v_ode, u_ode, semi) = system

function apply_spatial_criterion!(system,
                                  neighbor_system::Union{WallBoundarySystem,
                                                         AbstractStructureSystem,
                                                         OpenBoundarySystem},
                                  v_ode, u_ode, semi)
    (; target_spacing, is_fixed) = system.cache
    neighbor_spacing = neighbor_system.initial_condition.particle_spacing

    u = wrap_u(u_ode, system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_symmetric_point_neighbor(system, neighbor_system,
                                     system_coords, neighbor_coords, semi;
                                     points=each_integrated_particle(system)) do particle,
                                                                                 _, _, _
        # The first boundary in this call overwrites the target spacing of the previous
        # call, further boundaries can only decrease it (`MarkBoundary` in the code
        # of Haftu et al.). Note that each thread only writes to its own `particle`.
        if is_fixed[particle]
            target_spacing[particle] = min(target_spacing[particle], neighbor_spacing)
        else
            target_spacing[particle] = neighbor_spacing
            is_fixed[particle] = true
        end
    end

    return system
end

function apply_solution_criteria!(system, criteria, v_ode, u_ode, semi)
    (; target_spacing, is_fixed) = system.cache
    (; max_spacing) = get_refinement(system)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # Compute the data that the criteria need for all particles, like the vorticity
    criteria_data = map(criteria) do criterion
        return prepare_refinement_criterion!(criterion, system, v_ode, u_ode, semi)
    end

    @threaded semi for particle in each_integrated_particle(system)
        # Particles near boundaries keep the boundary spacing
        # (`SolutionAdaptive` in the code of Haftu et al.)
        is_fixed[particle] && return

        requested_spacings = map(criteria, criteria_data) do criterion, data
            return requested_target_spacing(criterion, data, system, particle, v, u, semi)
        end
        requested_spacing = reduce(min, requested_spacings)

        # The criteria are only active where they request a spacing below `max_spacing`
        requested_spacing < max_spacing || return

        # A solution criterion can only decrease the spacing of the previous call
        target_spacing[particle] = min(target_spacing[particle], requested_spacing)
        is_fixed[particle] = true
    end

    return system
end

# Returns the data that `requested_target_spacing` needs for all particles
@inline prepare_refinement_criterion!(criterion, system, v_ode, u_ode, semi) = nothing

# Spatial criteria are applied in `apply_spatial_criterion!`
@inline function requested_target_spacing(::SpatialRefinementCriterion, data, system,
                                          particle, v, u, semi)
    return typemax(eltype(system))
end

@inline function requested_target_spacing(criterion::SolutionRefinementCriterion, data,
                                          system, particle, v, u, semi)
    return criterion.target_spacing(system, particle, v, u, semi)
end

function prepare_refinement_criterion!(criterion::VorticityRefinementCriterion, system,
                                       v_ode, u_ode, semi)
    (; relative_threshold, absolute_threshold, outlet_cutoff) = criterion
    (; smoothed_vorticity) = system.cache
    ELTYPE = eltype(system)

    compute_vorticity!(system, v_ode, u_ode, semi)
    smooth_vorticity!(system, u_ode, semi)

    # Every particle is its own neighbor in `smooth_vorticity!`, so the largest smoothed
    # vorticity is the largest vorticity magnitude. Inactive particles have zero vorticity.
    # Note that the code of Haftu et al. uses the largest signed vorticity
    # (`numpy.max(dst.vor)` in `SmoothedVorticity`), which is too small for a flow
    # with mostly negative vorticity.
    vorticity_max = maximum(smoothed_vorticity)
    threshold = max(convert(ELTYPE, relative_threshold) * vorticity_max,
                    convert(ELTYPE, absolute_threshold))

    flow_direction = flow_direction_vector(criterion.flow_direction, system)
    outlet_position = outlet_position_along_flow(outlet_cutoff, flow_direction, system)
    spacing = vorticity_target_spacing(criterion.target_spacing, system)

    return (; threshold, flow_direction, outlet_position, target_spacing=spacing)
end

@inline function requested_target_spacing(::VorticityRefinementCriterion, data, system,
                                          particle, v, u, semi)
    (; threshold, flow_direction, outlet_position, target_spacing) = data
    (; smoothed_vorticity) = system.cache

    x = current_coords(u, system, particle)
    if smoothed_vorticity[particle] > threshold && dot(x, flow_direction) < outlet_position
        return target_spacing
    end

    return typemax(eltype(system))
end

@inline function vorticity_target_spacing(::Nothing, system)
    return get_refinement(system).min_spacing
end

@inline function vorticity_target_spacing(target_spacing, system)
    return convert(eltype(system), target_spacing)
end

# The default flow direction is the positive x-direction
@inline function flow_direction_vector(::Nothing, system)
    ELTYPE = eltype(system)

    return SVector(ntuple(i -> i == 1 ? one(ELTYPE) : zero(ELTYPE), Val(ndims(system))))
end

@inline function flow_direction_vector(flow_direction, system)
    return normalize(SVector{ndims(system), eltype(system)}(flow_direction))
end

@inline function outlet_position_along_flow(::Nothing, flow_direction, system)
    return typemax(eltype(system))
end

# Position of the outlet cutoff along the flow direction, as the fraction `outlet_cutoff`
# of the extent of the initial fluid along the flow direction.
# Note that this differs from the code of Haftu et al. (`xmax=0.9*xmax` in
# `AdaptiveResolutionVacondioNoBG`), which uses the fraction of the largest x-coordinate
# of the initial fluid. Both agree for a fluid that starts at `x = 0`. The fraction of the
# largest coordinate does not work for a fluid that does not start at zero, e.g.,
# for a flow in negative x-direction, where all coordinates along the flow are negative.
function outlet_position_along_flow(outlet_cutoff, flow_direction, system)
    (; initial_condition) = system

    # With a `SystemBuffer`, the buffer particles are appended to the initial condition
    # (see `allocate_buffer`) and placed far away from the domain, so skip them
    n_buffer = isnothing(buffer(system)) ? 0 : buffer(system).buffer_size
    n_initial = nparticles(initial_condition) - n_buffer
    coordinates = view(initial_condition.coordinates, :, 1:n_initial)

    # Position of the initial fluid particles along the flow direction.
    # Use broadcasting instead of a loop over the particles, which also works on GPUs.
    position_along_flow = sum(coordinates .* flow_direction, dims=1)
    position_min, position_max = extrema(position_along_flow)

    return convert(eltype(system),
                   position_min + outlet_cutoff * (position_max - position_min))
end

# The vorticity is a scalar in 2D and a vector in 3D
@inline n_vorticity_components(NDIMS) = NDIMS == 3 ? 3 : 1

# Compute the vorticity of all active particles
# (`Vorticity` in the code of Haftu et al.).
function compute_vorticity!(system, v_ode, u_ode, semi)
    set_zero!(system.cache.vorticity)

    # As in the code of Haftu et al. (`Vorticity` with `sources=fluids_with_io`),
    # walls and structures do not contribute to the vorticity, only fluids and open
    # boundaries. With the wall particles, the vorticity near no-slip walls would include
    # the wall shear and dominate the largest vorticity, so that the relative threshold
    # would refine less of the flow away from the walls.
    # The particles near walls are refined by the `SpatialRefinementCriterion` instead.
    foreach_system(semi) do neighbor_system
        add_vorticity!(system, neighbor_system, v_ode, u_ode, semi)
    end

    return system
end

@inline add_vorticity!(system, neighbor_system, v_ode, u_ode, semi) = system

function add_vorticity!(system,
                        neighbor_system::Union{AbstractFluidSystem, OpenBoundarySystem},
                        v_ode, u_ode, semi)
    (; vorticity) = system.cache
    n_components = n_vorticity_components(ndims(system))

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    # Each thread only writes to its own `particle`
    foreach_symmetric_point_neighbor(system, neighbor_system,
                                     system_coords, neighbor_coords, semi;
                                     points=each_integrated_particle(system)) do particle,
                                                                                 neighbor,
                                                                                 pos_diff,
                                                                                 distance
        # Skip the particle itself and neighbors with (almost) the same position,
        # where the kernel gradient would divide by zero (see `almostzero`)
        distance < almostzero(smoothing_length(system, particle)) && return

        # The average of the gradients with both smoothing lengths,
        # as in the code of Haftu et al.
        grad_kernel = symmetric_kernel_grad(system, neighbor_system,
                                            pos_diff, distance, particle, neighbor)

        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)
        v_a = @inbounds current_velocity(v, system, particle)
        v_b = @inbounds current_velocity(v_neighbor, neighbor_system, neighbor)

        vorticity_ = m_b / rho_b * vorticity_kernel_cross(grad_kernel, v_b - v_a)

        for i in 1:n_components
            @inbounds vorticity[(particle - 1) * n_components + i] += vorticity_[i]
        end
    end

    return system
end

# `∇W × (v_b - v_a)` for `∇ × v`. In 2D, only the z-component.
@inline function vorticity_kernel_cross(grad_kernel::SVector{2}, v_diff)
    return SVector(grad_kernel[1] * v_diff[2] - grad_kernel[2] * v_diff[1])
end

@inline function vorticity_kernel_cross(grad_kernel::SVector{3}, v_diff)
    return cross(grad_kernel, v_diff)
end

@propagate_inbounds function vorticity_magnitude(vorticity, particle,
                                                 ::Val{N}) where {N}
    vorticity_ = SVector(ntuple(i -> vorticity[(particle - 1) * N + i], Val(N)))

    return norm(vorticity_)
end

# The largest vorticity magnitude in the neighborhood of each active particle,
# including the particle itself (`SmoothedVorticity` in the code of Haftu et al.).
# Only the particles of the same system are considered, as in the code of Haftu et al.
#
# Note that this differs from `SmoothedVorticity` in `no_bg.py` of Haftu et al.,
# which overwrites `vor_max` for every neighbor instead of taking the maximum, so that
# its result is `max(ω_a, |ω_b|)` with the last neighbor `b`. This implements the intended
# maximum over all neighbors, as in `SmoothedVorticity` in `adapt.py`.
function smooth_vorticity!(system, u_ode, semi)
    (; vorticity, smoothed_vorticity) = system.cache
    n_components = Val(n_vorticity_components(ndims(system)))

    # Inactive particles keep zero vorticity
    set_zero!(smoothed_vorticity)

    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)
    neighborhood_search = get_neighborhood_search(system, semi)
    backend = semi.parallelization_backend
    init = zero(eltype(system))

    @threaded semi for particle in each_integrated_particle(system)
        smoothed_vorticity[particle] = mapreduce_symmetric_neighbor(max, system, system,
                                                                    system_coords,
                                                                    system_coords,
                                                                    neighborhood_search,
                                                                    backend, particle;
                                                                    init) do _, neighbor,
                                                                             _, _
            return @inbounds vorticity_magnitude(vorticity, neighbor, n_components)
        end
    end

    return system
end
