# Hydrodynamic force on a wall boundary system (e.g. a cylinder) by momentum exchange.
#
# The force on the wall is the negative of the force that the wall exerts on the fluid.
# The pair force uses the same functions as the fluid-wall interaction of the solver
# (the pressure term of `add_momentum_equation` and `add_dv_viscosity`, without the extra
# terms of shifting techniques), so variable smoothing lengths and `beta` are included
# as in the simulation. This is the discrete form of Eq. 21 in Haftu et al.
#
# Exception: the pressures are not reduced by the average pressure of the neighbors of
# the fluid particle (`average_pressure_reduction` of EDAC, on by default with a shifting
# technique). In the solver, the reduced pair force is not antisymmetric, so its wall
# share is not the force on the wall: near the front stagnation point p ≈ p_avg, and the
# pressure drag of a cylinder at Re = 100 came out as Cd_p ≈ -0.1 instead of ≈ 1.
# For each fluid particle, the reduction only redistributes the pressure force among its
# neighbors (Σ_b V_b ∇W_ab ≈ 0 with full support), so the plain pressures give the
# physical force.
#
# The viscous part is `add_dv_viscosity` alone, and the pressure part is the rest.

using TrixiParticles
using TrixiParticles: SVector

const TP = TrixiParticles

"""
    wall_force(fluid_system, wall_system, v_ode, u_ode, semi)

Return `(; total, pressure, viscous)` as `SVector`s: the force of `fluid_system` on
`wall_system`. Only valid for an `EntropicallyDampedSPHSystem` as `fluid_system`.
Call `TrixiParticles.update_systems_and_nhs` before, unless this is called from a
`PostprocessCallback`, which does that already.
"""
function wall_force(fluid_system::EntropicallyDampedSPHSystem, wall_system,
                    v_ode, u_ode, semi)
    v_fluid = TP.wrap_v(v_ode, fluid_system, semi)
    u_fluid = TP.wrap_u(u_ode, fluid_system, semi)
    v_wall = TP.wrap_v(v_ode, wall_system, semi)
    u_wall = TP.wrap_u(u_ode, wall_system, semi)

    NDIMS = ndims(fluid_system)
    ELTYPE = eltype(fluid_system)
    (; sound_speed) = fluid_system

    # Acceleration of each fluid particle due to the wall. Each fluid particle is
    # processed by one thread, so writing to its column is thread-safe.
    dv_total = zeros(ELTYPE, NDIMS, TP.nparticles(fluid_system))
    dv_viscous = zeros(ELTYPE, NDIMS, TP.nparticles(fluid_system))

    system_coords = TP.current_coordinates(u_fluid, fluid_system)
    neighbor_coords = TP.current_coordinates(u_wall, wall_system)
    h = TP.initial_smoothing_length(fluid_system)

    TP.foreach_symmetric_point_neighbor(fluid_system, wall_system,
                                        system_coords, neighbor_coords, semi;
                                        points=TP.each_integrated_particle(fluid_system)) do particle,
                                                                                             neighbor,
                                                                                             pos_diff,
                                                                                             distance
        # Same as in `interact!` of the EDAC system
        TP.skip_zero_distance(fluid_system) && distance < TP.almostzero(h) && return

        grad_kernel = TP.symmetric_kernel_grad(fluid_system, wall_system,
                                               pos_diff, distance, particle, neighbor)

        rho_a = TP.current_density(v_fluid, fluid_system, particle)
        rho_b = TP.current_density(v_wall, wall_system, neighbor)
        v_a = TP.current_velocity(v_fluid, fluid_system, particle)
        v_b = TP.current_velocity(v_wall, wall_system, neighbor)
        p_a = TP.current_pressure(v_fluid, fluid_system, particle)
        p_b = TP.current_pressure(v_wall, wall_system, neighbor)
        m_a = TP.hydrodynamic_mass(fluid_system, particle)
        m_b = TP.hydrodynamic_mass(wall_system, neighbor)

        # Pressure term of `add_momentum_equation` with the plain pressures, i.e.,
        # without the average pressure reduction (see the header of this file)
        beta_a = TP.get_beta(fluid_system, particle)
        beta_b = TP.get_beta(wall_system, neighbor)
        dv_pressure = TP.pressure_acceleration_variable_h(fluid_system, wall_system,
                                                          particle, neighbor, m_a, m_b,
                                                          p_a / beta_a, p_b / beta_b,
                                                          rho_a, rho_b, pos_diff, distance,
                                                          grad_kernel,
                                                          fluid_system.correction)
        dv_visc = TP.add_dv_viscosity(zero(v_a), fluid_system, wall_system,
                                      v_fluid, v_wall, particle, neighbor,
                                      pos_diff, distance, sound_speed,
                                      m_a, m_b, rho_a, rho_b, v_a, v_b, grad_kernel)

        for i in 1:NDIMS
            dv_total[i, particle] += dv_pressure[i] + dv_visc[i]
            dv_viscous[i, particle] += dv_visc[i]
        end
    end

    # Newton's third law: the wall experiences the opposite force
    total = zero(SVector{NDIMS, ELTYPE})
    viscous = zero(SVector{NDIMS, ELTYPE})
    for particle in TP.each_integrated_particle(fluid_system)
        m_a = TP.hydrodynamic_mass(fluid_system, particle)
        total -= m_a * SVector{NDIMS}(view(dv_total, :, particle))
        viscous -= m_a * SVector{NDIMS}(view(dv_viscous, :, particle))
    end

    return (; total, pressure=total - viscous, viscous)
end

"""
    ForceCoefficients(semi, fluid_system, wall_system; reference_density,
                      reference_velocity, reference_length, flow_direction=(1, 0))

Custom quantities for the `PostprocessCallback` that write the drag and lift coefficients
(Eq. 22 in Haftu et al.) of a wall system, total and split into pressure and viscous parts.
The systems are identified by their index in the semidiscretization (e.g., the order in
`Semidiscretization(fluid_system, open_boundary, wall_system, cylinder_system)`), so that
the callback can be created before the semidiscretization:

```julia
coefficients = ForceCoefficients(; fluid_index=1, wall_index=4, ...)
PostprocessCallback(; dt=0.01, quantities(coefficients)...)
```

The lift direction is `flow_direction` rotated by 90° counterclockwise.
The force is computed once per time and shared by all quantities.
"""
mutable struct ForceCoefficients
    fluid_index    :: Int
    wall_index     :: Int
    drag_direction :: SVector{2, Float64}
    lift_direction :: SVector{2, Float64}
    normalization  :: Float64
    last_time      :: Float64
    last_force     :: Any
end

function ForceCoefficients(; fluid_index, wall_index, reference_density,
                           reference_velocity, reference_length, flow_direction=(1, 0))
    drag_direction = TP.normalize(SVector{2, Float64}(flow_direction...))
    lift_direction = SVector(-drag_direction[2], drag_direction[1])
    normalization = 0.5 * reference_density * reference_velocity^2 * reference_length

    return ForceCoefficients(fluid_index, wall_index, drag_direction, lift_direction,
                             normalization, NaN, nothing)
end

function current_force!(coefficients::ForceCoefficients, v_ode, u_ode, semi, t)
    if t != coefficients.last_time
        fluid_system = semi.systems[coefficients.fluid_index]
        wall_system = semi.systems[coefficients.wall_index]

        coefficients.last_force = wall_force(fluid_system, wall_system, v_ode, u_ode, semi)
        coefficients.last_time = t
    end

    return coefficients.last_force
end

function coefficient_quantity(coefficients, part, direction)
    return function (system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        # Only write one column, for the fluid system
        TP.system_indices(system, semi) == coefficients.fluid_index || return nothing

        force = getproperty(current_force!(coefficients, v_ode, u_ode, semi, t), part)

        return TP.dot(force, getproperty(coefficients, direction)) /
               coefficients.normalization
    end
end

# `prefix` distinguishes several bodies, e.g. `prefix="upper_"` gives `upper_cd`.
# Note that the `PostprocessCallback` appends `_<system name>` to each column name.
function quantities(coefficients::ForceCoefficients; prefix="")
    parts = (cd=(:total, :drag_direction), cl=(:total, :lift_direction),
             cd_pressure=(:pressure, :drag_direction),
             cl_pressure=(:pressure, :lift_direction),
             cd_viscous=(:viscous, :drag_direction),
             cl_viscous=(:viscous, :lift_direction))

    names = Tuple(Symbol(prefix, name) for name in keys(parts))
    functions = Tuple(coefficient_quantity(coefficients, part...) for part in values(parts))

    return NamedTuple{names}(functions)
end
