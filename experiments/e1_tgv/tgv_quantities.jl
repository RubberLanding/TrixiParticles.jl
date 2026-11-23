# Custom quantities of the Taylor-Green vortex for the `PostprocessCallback`.
#
# Errors against the exact solution, weighted with the particle volume m/ρ, so that
# particles of different sizes count by the area they represent:
#   e_L1 = Σ V_a |φ_a - φ_exact(x_a)| / Σ V_a |φ_exact(x_a)|
#   e_L2 = sqrt(Σ V_a |φ_a - φ_exact(x_a)|² / Σ V_a |φ_exact(x_a)|²)
# for the velocity (vector difference) and the pressure.
# The EDAC pressure has no fixed level, so the volume-weighted mean pressure is subtracted
# (the exact pressure has zero mean over the periodic box). The pressure errors are
# normalized by the largest exact pressure instead, since the exact pressure is zero
# on lines through the domain.
# The `_box` variants only use the particles inside the refinement box.

using TrixiParticles
using CSV
using DataFrames

const TP = TrixiParticles

struct TaylorGreenVortex
    U::Float64
    b::Float64
    box_min::NTuple{2, Float64}
    box_max::NTuple{2, Float64}
end

function TaylorGreenVortex(; U=1.0, reynolds_number=100.0, box_min=(0.35, 0.35),
                           box_max=(0.65, 0.65))
    return TaylorGreenVortex(U, -8pi^2 / reynolds_number, box_min, box_max)
end

function exact_velocity(tgv::TaylorGreenVortex, x, t)
    return tgv.U * exp(tgv.b * t) *
           TP.SVector(-cos(2pi * x[1]) * sin(2pi * x[2]),
                      sin(2pi * x[1]) * cos(2pi * x[2]))
end

function exact_pressure(tgv::TaylorGreenVortex, x, t)
    return -tgv.U^2 * exp(2 * tgv.b * t) * (cos(4pi * x[1]) + cos(4pi * x[2])) / 4
end

# Exact kinetic energy ∫ ρ |u|² / 2 over the unit box with ρ = 1
exact_kinetic_energy(tgv::TaylorGreenVortex, t) = tgv.U^2 / 4 * exp(2 * tgv.b * t)

function in_box(tgv::TaylorGreenVortex, x)
    return tgv.box_min[1] < x[1] < tgv.box_max[1] && tgv.box_min[2] < x[2] < tgv.box_max[2]
end

# All error sums in one pass over the particles
function error_sums(tgv, system, v_ode, u_ode, semi, t; only_box)
    v = TP.wrap_v(v_ode, system, semi)
    u = TP.wrap_u(u_ode, system, semi)
    particles = TP.each_active_particle(system)

    volume(particle) = TP.hydrodynamic_mass(system, particle) /
                       TP.current_density(v, system, particle)

    total_volume = sum(volume, particles)
    p_mean = sum(particle -> volume(particle) *
                             TP.current_pressure(v, system, particle),
                 particles) / total_volume

    sums = zeros(6)
    p_max_exact = 0.0
    for particle in particles
        x = TP.current_coords(u, system, particle)
        only_box && !in_box(tgv, x) && continue

        V = volume(particle)
        v_exact = exact_velocity(tgv, x, t)
        v_diff = TP.current_velocity(v, system, particle) - v_exact
        p_exact = exact_pressure(tgv, x, t)
        p_diff = TP.current_pressure(v, system, particle) - p_mean - p_exact

        sums[1] += V * TP.norm(v_diff)
        sums[2] += V * TP.norm(v_exact)
        sums[3] += V * TP.dot(v_diff, v_diff)
        sums[4] += V * TP.dot(v_exact, v_exact)
        sums[5] += V * abs(p_diff)
        sums[6] += V * p_diff^2
        p_max_exact = max(p_max_exact, abs(p_exact))
    end

    # Volume over which the errors are summed (the box or the domain)
    volume_sum = only_box ?
                 sum(particle -> in_box(tgv, TP.current_coords(u, system, particle)) ?
                                 volume(particle) : 0.0, particles) :
                 total_volume

    return (; v_l1=sums[1] / sums[2], v_l2=sqrt(sums[3] / sums[4]),
            p_l1=sums[5] / (volume_sum * p_max_exact),
            p_l2=sqrt(sums[6] / volume_sum) / p_max_exact)
end

"""
    tgv_quantities(tgv; fluid_index=1)

Errors (`v_l1`, `v_l2`, `p_l1`, `p_l2` and the same with `_box`) and the kinetic energy
`kinetic_energy` (with `kinetic_energy_exact`) for the `PostprocessCallback`.
"""
function tgv_quantities(tgv; fluid_index=1)
    cache = Ref{Any}((t=NaN, values=nothing))

    function errors(v_ode, u_ode, semi, t)
        if cache[].t != t
            system = semi.systems[fluid_index]
            domain = error_sums(tgv, system, v_ode, u_ode, semi, t; only_box=false)
            box = error_sums(tgv, system, v_ode, u_ode, semi, t; only_box=true)
            cache[] = (; t, values=(; domain, box))
        end

        return cache[].values
    end

    function quantity(region, name)
        return function (system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            TP.system_indices(system, semi) == fluid_index || return nothing

            return getproperty(getproperty(errors(v_ode, u_ode, semi, t), region), name)
        end
    end

    function kinetic_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing

        v = TP.wrap_v(v_ode, system, semi)
        return sum(particle -> TP.hydrodynamic_mass(system, particle) / 2 *
                               sum(abs2, TP.current_velocity(v, system, particle)),
                   TP.each_active_particle(system))
    end

    function kinetic_energy_exact(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing

        return exact_kinetic_energy(tgv, t)
    end

    return (; v_l1=quantity(:domain, :v_l1), v_l2=quantity(:domain, :v_l2),
            p_l1=quantity(:domain, :p_l1), p_l2=quantity(:domain, :p_l2),
            v_l1_box=quantity(:box, :v_l1), v_l2_box=quantity(:box, :v_l2),
            p_l1_box=quantity(:box, :p_l1), p_l2_box=quantity(:box, :p_l2),
            kinetic_energy, kinetic_energy_exact)
end

"""
    grid_snapshot_quantity(tgv, output_directory, times; resolution, fluid_index=1)

A `PostprocessCallback` quantity that writes the fields interpolated on a regular grid
with the exact solution to `grid_t<time>.csv` at each of `times`, and returns `nothing`
(no column in the metrics). The `PostprocessCallback` must be evaluated at these times,
i.e., they must be multiples of its `dt`.
"""
function grid_snapshot_quantity(tgv, output_directory, times; resolution, fluid_index=1)
    function snapshot(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing
        any(time -> isapprox(t, time; atol=1e-8), times) || return nothing

        # The grid points are at the cell centers, so that no point lies on the
        # periodic boundary
        result = interpolate_plane_2d((resolution / 2, resolution / 2),
                                      (1 - resolution / 2, 1 - resolution / 2),
                                      resolution, semi, system, v_ode, u_ode;
                                      cut_off_bnd=false)

        points = result.point_coords
        x, y = points[1, :], points[2, :]
        v_exact = [exact_velocity(tgv, points[:, i], t) for i in axes(points, 2)]

        df = DataFrame(; x, y, vx=result.velocity[1, :], vy=result.velocity[2, :],
                       p=result.pressure, density=result.density,
                       vx_exact=getindex.(v_exact, 1), vy_exact=getindex.(v_exact, 2),
                       p_exact=[exact_pressure(tgv, points[:, i], t)
                                for i in axes(points, 2)])

        mkpath(output_directory)
        CSV.write(joinpath(output_directory, "grid_t$(round(t, digits=4)).csv"), df)

        return nothing
    end

    return (; grid_snapshot=snapshot)
end
