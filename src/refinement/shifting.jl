"""
    ShiftingHaftu(; max_iterations=100, tolerance=2e-3, theta=0.5, max_shift_factor=0.05)

Shifting of the particles after splitting and merging by Haftu et al. (Eq. 15 and 36),
to be passed as `shifting_technique` to `ParticleRefinement`.
The particle properties are corrected with a Taylor series expansion (Eq. 37).

The particles are shifted iteratively. In each iteration, they are shifted by `theta`
times the shift of Eq. 15, but by at most `max_shift_factor` times their smoothing length.
The iteration stops when the largest shift of Eq. 15 is smaller than `tolerance` times
the smoothing length of its particle, or after `max_iterations` iterations.
A particle is not shifted in an iteration if it would end up in an open boundary zone
of its fluid system, where it would be converted to an open boundary particle.
The code of Haftu et al. instead uses three iterations with `theta=1` and a maximum shift
of `0.00625` times the smoothing length.

Velocity and pressure are corrected for all density calculators. The density is only
corrected with [`ContinuityDensity`](@ref), since [`SummationDensity`](@ref) computes
the density from the shifted positions before it is used again.
"""
struct ShiftingHaftu{ELTYPE}
    max_iterations   :: Int
    tolerance        :: ELTYPE
    theta            :: ELTYPE
    max_shift_factor :: ELTYPE
end

function ShiftingHaftu(; max_iterations=100, tolerance=2e-3, theta=0.5,
                       max_shift_factor=0.05)
    return ShiftingHaftu(max_iterations, promote(tolerance, theta, max_shift_factor)...)
end

"""
    shift_particles!(semi, v_ode, u_ode)

Shift the particles of fluid systems with particle refinement after splitting and merging
with the `shifting_technique` of their `ParticleRefinement` (see [`ShiftingHaftu`](@ref)).
After shifting, the smoothing lengths are updated with the average mass of the neighbors
(Eq. 35 in Haftu et al.). This is also done with `shifting_technique=nothing`.
"""
function shift_particles!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        shift_particles!(system, v_ode, u_ode, semi)
    end

    # Update all neighborhood searches that contain the shifted particles
    @trixi_timeit timer() "update nhs" update_nhs!(semi, u_ode)

    return semi
end

@inline shift_particles!(system, v_ode, u_ode, semi) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return shift_particles!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi) = system

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi)
    return apply_refinement_shifting!(system, refinement.shifting_technique,
                                      v_ode, u_ode, semi)
end

# Without shifting, only the smoothing lengths are updated to match the masses
# after splitting and merging
function apply_refinement_shifting!(system, ::Nothing, v_ode, u_ode, semi)
    u = wrap_u(u_ode, system, semi)
    @trixi_timeit timer() "smoothing lengths after" update_smoothing_lengths!(system, u,
                                                                              semi)

    return system
end

function apply_refinement_shifting!(system, shifting::ShiftingHaftu, v_ode, u_ode, semi)
    (; total_shift) = system.cache
    (; max_iterations) = shifting
    ELTYPE = eltype(system)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    tolerance = convert(ELTYPE, shifting.tolerance)
    theta = convert(ELTYPE, shifting.theta)
    max_shift_factor = convert(ELTYPE, shifting.max_shift_factor)

    set_zero!(total_shift)

    # Shift with smoothing lengths that match the masses after splitting and merging
    @trixi_timeit timer() "smoothing lengths before" update_smoothing_lengths!(system, u,
                                                                               semi)

    for _ in 1:max_iterations
        @trixi_timeit timer() "compute shift" compute_shift!(system, u, u_ode, semi)
        max_relative_shift = @trixi_timeit timer() "apply shift" begin
            apply_shift!(system, u, semi, theta, max_shift_factor)
        end

        max_relative_shift < tolerance && break
    end

    @trixi_timeit timer() "correct properties" begin
        correct_properties!(system, v, u, v_ode, u_ode, semi)
    end

    # Smoothing lengths for the shifted positions
    @trixi_timeit timer() "smoothing lengths after" update_smoothing_lengths!(system, u,
                                                                              semi)

    return system
end

# The shift of Eq. 15 in Haftu et al. As in their code, all systems are included,
# so that particles are also shifted away from boundaries.
@fastpow function compute_shift!(system, u, u_ode, semi)
    (; shift) = system.cache
    (; reference_density) = get_refinement(system)
    smoothing_kernel = system_smoothing_kernel(system)
    xi = convert(eltype(system), kernel_inflection_point(smoothing_kernel))

    set_zero!(shift)

    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        has_system_interaction(system, neighbor_system, semi) || return

        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            h_a, h_b = pair_smoothing_lengths(system, neighbor_system, particle, neighbor)
            h_ab = (h_a + h_b) / 2
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            W = kernel(smoothing_kernel, distance, h_ab)
            W_xi = kernel(smoothing_kernel, xi * h_ab, h_ab)
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

            # Eq. 15
            shift_ = -h_a^2 / 2 * m_b / reference_density *
                     (1 + 24 * (W / W_xi)^4 / 100) * grad_kernel

            @inbounds add_svector_flat!(shift, shift_, particle)
        end
    end

    return system
end

# Shift the particles by `theta` times the shift of Eq. 15, but by at most
# `max_shift_factor` times their smoothing length. Particles that would end up in an open
# boundary zone are not shifted (see `reject_shift_into_boundary_zones!`).
# Return the largest shift of Eq. 15 divided by the smoothing length of its particle.
function apply_shift!(system, u, semi, theta, max_shift_factor)
    (; shift, total_shift, relative_shift) = system.cache
    NDIMS = ndims(system)

    set_zero!(relative_shift)

    # Limit the shift. Afterwards, `shift` contains the shift that is applied.
    @trixi_timeit timer() "limit" begin
        @threaded semi for particle in each_integrated_particle(system)
            shift_a = extract_svector_flat(shift, Val(ndims(system)), particle)
            shift_norm = norm(shift_a)
            h_a = smoothing_length(system, particle)

            max_shift = max_shift_factor * h_a
            factor = theta * shift_norm > max_shift ? max_shift / shift_norm : theta
            for dim in 1:NDIMS
                shift[linear_index_vector(dim, particle, NDIMS)] = factor * shift_a[dim]
            end

            relative_shift[particle] = shift_norm / h_a
        end
    end

    @trixi_timeit timer() "reject shift into zones" begin
        foreach_system(semi) do neighbor_system
            reject_shift_into_boundary_zones!(system, neighbor_system, u, semi)
        end
    end

    @trixi_timeit timer() "apply" begin
        @threaded semi for particle in each_integrated_particle(system)
            applied_shift = extract_svector_flat(shift, Val(ndims(system)), particle)

            position = current_coords(u, system, particle) + applied_shift
            set_particle_position!(u, system, particle, position)
            add_svector_flat!(total_shift, applied_shift, particle)
        end
    end

    # Only the self-interaction neighborhood search is used in the next iteration
    neighborhood_search = get_neighborhood_search(system, semi)
    @trixi_timeit timer() "update self nhs" begin
        update_nhs!(neighborhood_search, system, system, u, u, semi)
    end

    return @trixi_timeit timer() "max shift" maximum(relative_shift)
end

@inline reject_shift_into_boundary_zones!(system, neighbor_system, u, semi) = system

# Do not shift a particle into an open boundary zone of its fluid system, where the open
# boundary would convert it to an open boundary particle. At an inflow, this adds mass:
# a released fluid particle is a copy of the open boundary particle that crossed the face,
# and that particle stays in the zone, moved back by the zone width. If the copy is shifted
# back into the zone, it becomes an additional open boundary particle that releases
# particles itself. The inflow then releases more mass than prescribed.
function reject_shift_into_boundary_zones!(system, neighbor_system::OpenBoundarySystem,
                                           u, semi)
    # Only the fluid system of the open boundary converts particles
    neighbor_system.fluid_system_index[] == system_indices(system, semi) || return system

    (; shift, relative_shift) = system.cache
    (; boundary_zones) = neighbor_system
    NDIMS = ndims(system)

    @threaded semi for particle in each_integrated_particle(system)
        shift_a = extract_svector_flat(shift, Val(ndims(system)), particle)
        new_position = current_coords(u, system, particle) + shift_a

        foreach_noalloc(boundary_zones) do boundary_zone
            if is_in_boundary_zone(boundary_zone, new_position)
                for dim in 1:NDIMS
                    shift[linear_index_vector(dim, particle, NDIMS)] = zero(eltype(system))
                end
                relative_shift[particle] = 0
            end
        end
    end

    return system
end

# Correct the particle properties after shifting with a Taylor series expansion (Eq. 37),
# `ϕ_a' = ϕ_a + (∇ϕ)_a ⋅ δr_a` with `(∇ϕ)_a = ∑_b V_b (ϕ_b - ϕ_a) ∇W_ab`, where `δr_a`
# is the total shift of all iterations. As in the code of Haftu et al., the gradients are
# computed at the shifted positions and all systems are included.
# Unlike in their code, free-slip walls do not contribute to the velocity gradient
# (see `correction_velocity`).
function correct_properties!(system, v, u, v_ode, u_ode, semi)
    (; total_shift, velocity_correction, pressure_correction,
     density_correction) = system.cache
    smoothing_kernel = system_smoothing_kernel(system)
    NDIMS = ndims(system)

    set_zero!(velocity_correction)
    set_zero!(pressure_correction)
    set_zero!(density_correction)

    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        has_system_interaction(system, neighbor_system, semi) || return

        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            h_a, h_b = pair_smoothing_lengths(system, neighbor_system, particle, neighbor)
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, (h_a + h_b) / 2)

            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = @inbounds current_density(v, system, particle)
            rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)
            v_a = @inbounds current_velocity(v, system, particle)
            v_b = @inbounds correction_velocity(v_neighbor, neighbor_system, neighbor, v_a)
            p_a = @inbounds current_pressure(v, system, particle)
            p_b = @inbounds current_pressure(v_neighbor, neighbor_system, neighbor)

            shift_a = @inbounds extract_svector_flat(total_shift, Val(ndims(system)),
                                                     particle)
            weight = m_b / rho_b * dot(grad_kernel, shift_a)

            @inbounds add_svector_flat!(velocity_correction, weight * (v_b - v_a),
                                        particle)
            @inbounds pressure_correction[particle] += weight * (p_b - p_a)
            @inbounds density_correction[particle] += weight * (rho_b - rho_a)
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        velocity = current_velocity(v, system, particle) +
                   extract_svector_flat(velocity_correction, Val(ndims(system)), particle)
        pressure = current_pressure(v, system, particle) + pressure_correction[particle]

        set_particle_velocity!(v, system, particle, velocity)
        set_particle_pressure!(v, system, particle, pressure)
        correct_density!(v, system, system.density_calculator, particle,
                         density_correction[particle])
    end

    return system
end

# The neighbor velocity `v_b` in the velocity correction of Eq. 37.
#
# Free-slip walls (wall boundary models without viscosity) return `v_a`, so that they do not
# contribute to the velocity correction. At a free-slip wall, the tangential velocity
# of the fluid does not change towards the wall. The wall particles, however, carry
# the wall velocity (zero for a resting wall). With the wall velocity, `v_b - v_a` would
# create a velocity gradient of about `|v_a| / h` normal to the wall, which does not exist
# in the flow. A particle shifted towards the wall would then be slowed down by this
# gradient, as if the wall were a no-slip wall. In the vortex street, the shift moves
# the first fluid rows slightly towards the walls in every refinement call, which slowed
# them down by more than 10 %.
#
# No-slip walls (with viscosity) use the wall velocity, because the velocity gradient
# towards these walls is physical. All other systems use their particle velocity.
# The code of Haftu et al. uses the wall velocity for all walls,
# but their test cases have no free-slip walls.
@propagate_inbounds function correction_velocity(v_neighbor, neighbor_system, neighbor,
                                                 v_a)
    return current_velocity(v_neighbor, neighbor_system, neighbor)
end

@propagate_inbounds function correction_velocity(v_neighbor,
                                                 neighbor_system::WallBoundarySystem,
                                                 neighbor, v_a)
    return correction_velocity(v_neighbor, neighbor_system,
                               neighbor_system.boundary_model.viscosity, neighbor, v_a)
end

# Free-slip wall
@inline function correction_velocity(v_neighbor, neighbor_system, ::Nothing, neighbor, v_a)
    return v_a
end

# No-slip wall
@propagate_inbounds function correction_velocity(v_neighbor, neighbor_system, viscosity,
                                                 neighbor, v_a)
    return current_velocity(v_neighbor, neighbor_system, neighbor)
end

@inline function correct_density!(v, system, ::ContinuityDensity, particle, correction)
    density = current_density(v, system, particle) + correction
    set_particle_density!(v, system, particle, density)

    return system
end

# The summation density is computed from the shifted positions before it is used again
@inline correct_density!(v, system, density_calculator, particle, correction) = system

# Update the smoothing lengths with the average mass of the neighbors (Eq. 35).
# As in `neighborhood_spacing`, only the particles of the same system within the compact
# support of `particle` are considered, including the particle itself.
function update_smoothing_lengths!(system, u, semi)
    (; reference_density) = get_refinement(system)
    smoothing_kernel = system_smoothing_kernel(system)
    ELTYPE = eltype(system)

    system_coords = current_coordinates(u, system)
    neighborhood_search = get_neighborhood_search(system, semi)

    # Sum of the masses and number of neighbors
    init = (zero(ELTYPE), 0)
    op(a, b) = (a[1] + b[1], a[2] + b[2])

    # Each thread only reads and writes the smoothing length of its own particle
    @threaded semi for particle in each_integrated_particle(system)
        radius = compact_support(smoothing_kernel, smoothing_length(system, particle))

        mass_sum,
        n_neighbors = mapreduce_neighbor(op, system_coords, system_coords,
                                         neighborhood_search,
                                         semi.parallelization_backend,
                                         particle; init) do particle, neighbor, _, distance
            distance > radius && return init

            return (hydrodynamic_mass(system, neighbor), 1)
        end

        # The particle itself is always a neighbor
        mass_avg = mass_sum / n_neighbors
        spacing = (mass_avg / reference_density)^(1 / ndims(system))
        set_particle_smoothing_length!(system, particle,
                                       spacing_to_smoothing_length(system, spacing))
    end

    return system
end

# The point of inflection `ξ` of the kernel in units of the smoothing length,
# where the second derivative of the kernel is zero (see Eq. 15 in Haftu et al.)
function kernel_inflection_point(smoothing_kernel)
    throw(ArgumentError("shifting with particle refinement is not supported for the " *
                        "kernel `$(nameof(typeof(smoothing_kernel)))`"))
end

@inline kernel_inflection_point(::SchoenbergCubicSplineKernel) = 2 / 3
@inline kernel_inflection_point(::SchoenbergQuinticSplineKernel) = 0.759298480738450
@inline kernel_inflection_point(::WendlandC2Kernel) = 1 / 2
