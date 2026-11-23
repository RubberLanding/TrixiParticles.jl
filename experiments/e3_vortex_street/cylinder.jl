# Setup helpers and snapshot quantities of the single-cylinder vortex street (E3).

using TrixiParticles
using CSV
using DataFrames

const TP = TrixiParticles

"""
    ring_fluid(box_fluid, particle_spacing, radius, center, density; ring_thickness,
               velocity)

The fluid `box_fluid` (a Cartesian lattice with `particle_spacing`) with concentric rings
of fluid particles around the cylinder of `radius` instead of the lattice cut out by the
cylinder. The rings reach from `radius` to `radius + ring_thickness`, and the lattice
particles are removed there.

Cutting the cylinder out of the lattice (`setdiff(box_fluid, cylinder)`) leaves an
irregular gap between the fluid and the cylinder surface, which gave a non-converging
~10 % error of the force in the Archimedes check (`common/test_forces.jl`).
With rings, the force converged at first order.
"""
function ring_fluid(box_fluid, particle_spacing, radius, center, density; ring_thickness,
                    velocity)
    n_layers = max(1, round(Int, ring_thickness / particle_spacing))

    rings = SphereShape(particle_spacing, radius, center, density;
                        sphere_type=RoundSphere(), n_layers, layer_outwards=true, velocity)
    disk = SphereShape(particle_spacing, radius + n_layers * particle_spacing, center,
                       density; sphere_type=RoundSphere())

    return union(rings, setdiff(box_fluid, disk))
end

# Vorticity ω_a = Σ_b V_b ∇W_ab × (v_b - v_a) of all active fluid particles (2D, scalar),
# with the same kernel gradient as in the solver (also with variable smoothing lengths)
function particle_vorticity(system, v_ode, u_ode, semi)
    v = TP.wrap_v(v_ode, system, semi)
    u = TP.wrap_u(u_ode, system, semi)
    coords = TP.current_coordinates(u, system)
    vorticity = zeros(TP.nparticles(system))

    # The symmetric loop skips pairs outside of the support of their smoothing lengths,
    # as in the solver
    TP.foreach_symmetric_point_neighbor(system, system, coords, coords, semi;
                                        points=TP.each_active_particle(system)) do particle,
                                                                                   neighbor,
                                                                                   pos_diff,
                                                                                   distance
        distance < sqrt(eps()) && return

        grad_kernel = TP.symmetric_kernel_grad(system, system, pos_diff, distance,
                                               particle, neighbor)
        volume_b = TP.hydrodynamic_mass(system, neighbor) /
                   TP.current_density(v, system, neighbor)
        v_diff = TP.current_velocity(v, system, neighbor) -
                 TP.current_velocity(v, system, particle)

        # ∇W_ab × (v_b - v_a), where ∇W_ab is the gradient with respect to x_a, so that
        # ω = ∂v/∂x - ∂u/∂y with the usual SPH gradient Σ_b V_b (f_b - f_a) ∇W_ab
        vorticity[particle] += volume_b *
                               (grad_kernel[1] * v_diff[2] - grad_kernel[2] * v_diff[1])
    end

    return vorticity
end

"""
    snapshot_quantity(output_directory, times; grid_min, grid_max, grid_resolution,
                      fluid_index=1)

A `PostprocessCallback` quantity that writes, at each of `times`,
- `particles_t<time>.csv`: position, velocity, pressure, mass, smoothing length and
  vorticity of all active fluid particles (E3c/E3d), and
- `grid_t<time>.csv`: velocity and pressure interpolated on a regular grid between
  `grid_min` and `grid_max` (E3e, the vorticity is computed from it in the postprocessing).
Returns `nothing` (no column in the metrics).
"""
function snapshot_quantity(output_directory, times; grid_min, grid_max, grid_resolution,
                           fluid_index=1)
    function snapshot(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
        TP.system_indices(system, semi) == fluid_index || return nothing
        any(time -> isapprox(t, time; atol=1e-8), times) || return nothing

        v = TP.wrap_v(v_ode, system, semi)
        u = TP.wrap_u(u_ode, system, semi)
        particles = collect(TP.each_active_particle(system))
        vorticity = particle_vorticity(system, v_ode, u_ode, semi)

        coords = [TP.current_coords(u, system, particle) for particle in particles]
        velocity = [TP.current_velocity(v, system, particle) for particle in particles]
        particle_table = DataFrame(; x=getindex.(coords, 1), y=getindex.(coords, 2),
                                   vx=getindex.(velocity, 1), vy=getindex.(velocity, 2),
                                   p=[TP.current_pressure(v, system, particle)
                                      for particle in particles],
                                   mass=[TP.hydrodynamic_mass(system, particle)
                                         for particle in particles],
                                   h=[TP.smoothing_length(system, particle)
                                      for particle in particles],
                                   vorticity=vorticity[particles])

        result = interpolate_plane_2d(grid_min, grid_max, grid_resolution, semi, system,
                                      v_ode, u_ode)
        points = result.point_coords
        grid_table = DataFrame(; x=points[1, :], y=points[2, :],
                               vx=result.velocity[1, :], vy=result.velocity[2, :],
                               p=result.pressure)

        mkpath(output_directory)
        suffix = "t$(round(t, digits=4)).csv"
        CSV.write(joinpath(output_directory, "particles_$suffix"), particle_table)
        CSV.write(joinpath(output_directory, "grid_$suffix"), grid_table)

        return nothing
    end

    return (; snapshot)
end
