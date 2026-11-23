# Plane Poiseuille flow for E2 and its quantities for the `PostprocessCallback`.
#
# The channel 0 < y < H has no-slip walls. Inflow and initial velocity are the parabolic
# profile, so the exact solution holds in the whole channel from the start and the short
# channel of the example (L = 1, entrance length ≈ 1.8 for a plug inflow at Re = 100) is
# enough:
#   u(y) = 4 U_max y (H - y) / H²,  v = 0,  dp/dx = -8 μ U_max / H² = -G
# With the outflow pressure 0 at x = L, the exact pressure is p(x) = G (L - x).
#
# Errors are evaluated in the measurement region x_min < x < x_max, away from the
# open boundaries, weighted with the particle volume m/ρ (see E1):
#   e_L1 = Σ V_a |v_a - v_exact(y_a)| / Σ V_a |v_exact(y_a)|
#   e_L2 = sqrt(Σ V_a |v_a - v_exact(y_a)|² / Σ V_a |v_exact(y_a)|²)
# The pressure gradient is the volume-weighted least-squares slope of p(x) of the particles
# in the region, which does not depend on the pressure level.

using TrixiParticles
using CSV
using DataFrames

const TP = TrixiParticles

struct PoiseuilleFlow
    H::Float64
    L::Float64
    U_max::Float64
    G::Float64          # -dp/dx
    region::NTuple{2, Float64}
end

function PoiseuilleFlow(; H, L, U_mean, kinematic_viscosity, density, region)
    U_max = 3 * U_mean / 2
    G = 8 * density * kinematic_viscosity * U_max / H^2

    return PoiseuilleFlow(H, L, U_max, G, region)
end

exact_velocity_x(flow::PoiseuilleFlow, y) = 4 * flow.U_max * y * (flow.H - y) / flow.H^2

function exact_velocity(flow::PoiseuilleFlow, x)
    return TP.SVector(exact_velocity_x(flow, x[2]), 0.0)
end

exact_pressure(flow::PoiseuilleFlow, x) = flow.G * (flow.L - x[1])

in_region(flow::PoiseuilleFlow, x) = flow.region[1] < x[1] < flow.region[2]

function error_sums(flow, system, v_ode, u_ode, semi)
    v = TP.wrap_v(v_ode, system, semi)
    u = TP.wrap_u(u_ode, system, semi)

    # v_l1 numerator/denominator, v_l2 numerator/denominator, and the sums of the
    # least-squares fit p = a + b x: Σ V, Σ V x, Σ V x², Σ V p, Σ V x p
    sums = zeros(9)
    for particle in TP.each_active_particle(system)
        x = TP.current_coords(u, system, particle)
        in_region(flow, x) || continue

        V = TP.hydrodynamic_mass(system, particle) / TP.current_density(v, system, particle)
        v_exact = exact_velocity(flow, x)
        v_diff = TP.current_velocity(v, system, particle) - v_exact
        p = TP.current_pressure(v, system, particle)

        sums[1] += V * TP.norm(v_diff)
        sums[2] += V * TP.norm(v_exact)
        sums[3] += V * TP.dot(v_diff, v_diff)
        sums[4] += V * TP.dot(v_exact, v_exact)
        sums[5] += V
        sums[6] += V * x[1]
        sums[7] += V * x[1]^2
        sums[8] += V * p
        sums[9] += V * x[1] * p
    end

    slope = (sums[5] * sums[9] - sums[6] * sums[8]) / (sums[5] * sums[7] - sums[6]^2)

    return (; v_l1=sums[1] / sums[2], v_l2=sqrt(sums[3] / sums[4]),
            dpdx=slope, dpdx_error=(slope + flow.G) / flow.G)
end

"""
    poiseuille_quantities(flow; fluid_index=1)

Errors `v_l1`, `v_l2` in the measurement region, the fitted pressure gradient `dpdx`,
its relative error `dpdx_error` and the exact value `dpdx_exact`.
"""
function poiseuille_quantities(flow; fluid_index=1)
    cache = Ref{Any}((t=NaN, values=nothing))

    function quantity(name)
        return function (system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            TP.system_indices(system, semi) == fluid_index || return nothing

            if cache[].t != t
                cache[] = (; t, values=error_sums(flow, system, v_ode, u_ode, semi))
            end

            return getproperty(cache[].values, name)
        end
    end

    function dpdx_exact(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing

        return -flow.G
    end

    return (; v_l1=quantity(:v_l1), v_l2=quantity(:v_l2), dpdx=quantity(:dpdx),
            dpdx_error=quantity(:dpdx_error), dpdx_exact)
end

"""
    profile_snapshot_quantity(flow, output_directory, times; stations, n_points,
                              n_points_centerline, fluid_index=1)

A `PostprocessCallback` quantity that writes, at each of `times`,
- `profiles_t<time>.csv`: the fields interpolated on vertical lines at the x `stations`,
  at `n_points` cell-centered points across the channel, with the exact velocity, and
- `centerline_t<time>.csv`: the fields on the centerline between the open boundaries,
  with the exact pressure.
Returns `nothing` (no column in the metrics).
"""
function profile_snapshot_quantity(flow, output_directory, times; stations, n_points,
                                   n_points_centerline, fluid_index=1)
    function snapshot(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing
        any(time -> isapprox(t, time; atol=1e-8), times) || return nothing

        dy = flow.H / n_points
        ys = [(i - 0.5) * dy for i in 1:n_points]
        points = reduce(hcat, [[x, y] for x in stations for y in ys])
        # With the wall velocity (the no-slip velocity of the dummy particles), the
        # interpolated profile has the boundary layer at the wall
        result = interpolate_points(points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                    include_wall_velocity=true)

        profiles = DataFrame(; x=points[1, :], y=points[2, :], vx=result.velocity[1, :],
                             vy=result.velocity[2, :], p=result.pressure,
                             density=result.density,
                             vx_exact=exact_velocity_x.(Ref(flow), points[2, :]))

        dx = flow.L / n_points_centerline
        xs = [(i - 0.5) * dx for i in 1:n_points_centerline]
        points = reduce(hcat, [[x, flow.H / 2] for x in xs])
        result = interpolate_points(points, semi, system, v_ode, u_ode; cut_off_bnd=false)

        centerline = DataFrame(; x=xs, vx=result.velocity[1, :], p=result.pressure,
                               density=result.density,
                               p_exact=[exact_pressure(flow, points[:, i])
                                        for i in axes(points, 2)])

        mkpath(output_directory)
        suffix = "t$(round(t, digits=4)).csv"
        CSV.write(joinpath(output_directory, "profiles_$suffix"), profiles)
        CSV.write(joinpath(output_directory, "centerline_$suffix"), centerline)

        return nothing
    end

    return (; profile_snapshot=snapshot)
end
