# Sanity check of `wall_force`: fluid at rest around a cylinder with the linear pressure
# field p = G x. The exact force on the cylinder is F = -∫ p n dA = (-G π R², 0)
# (the 2D Archimedes force), and the viscous part is zero.
#
# Run from the root of the TrixiParticles worktree:
#   julia --project=run -t 4 experiments/common/test_forces.jl

include(joinpath(@__DIR__, "forces.jl"))

function cylinder_force(particle_spacing; refinement=false, ring_fluid=false)
    fluid_density = 1000.0
    G = 100.0
    radius = 0.2
    center = (0.6, 0.6)
    box_length = 1.2

    smoothing_length_factor = 1.5
    smoothing_length = smoothing_length_factor * particle_spacing
    smoothing_kernel = WendlandC2Kernel{2}()

    n = round(Int, box_length / particle_spacing)
    box = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=fluid_density,
                           pressure=pos -> G * pos[1])
    cylinder = SphereShape(particle_spacing, radius, center, fluid_density;
                           sphere_type=RoundSphere())
    # Ring fluid: concentric layers outside the cylinder, consistent with its rings
    n_layers = round(Int, 0.4 / particle_spacing)
    fluid = ring_fluid ?
            SphereShape(particle_spacing, radius, center, fluid_density;
                        sphere_type=RoundSphere(), n_layers, layer_outwards=true,
                        pressure=pos -> G * pos[1]) :
            setdiff(box, cylinder)

    particle_refinement = refinement ?
                          ParticleRefinement(; spacing_ratio=1.2,
                                             min_spacing=particle_spacing / 2,
                                             max_spacing=particle_spacing,
                                             smoothing_length_factor,
                                             reference_density=fluid_density) : nothing

    viscosity = ViscosityAdami(nu=1e-3)
    fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                               sound_speed=10.0, viscosity,
                                               density_calculator=SummationDensity(),
                                               particle_refinement)

    boundary_model = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation=nothing, viscosity)
    cylinder_system = WallBoundarySystem(cylinder, boundary_model)

    semi = Semidiscretization(fluid_system, cylinder_system)
    ode = semidiscretize(semi, (0.0, 1.0))
    v_ode, u_ode = ode.u0.x
    TP.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    force = wall_force(semi.systems[1], semi.systems[2], v_ode, u_ode, semi)
    exact = -G * pi * radius^2

    return force, exact
end

println("dx       refinement  ring_fluid  F_x          F_y          F_x/exact  |F_viscous|")
for particle_spacing in (0.04, 0.02, 0.01), refinement in (false, true),
    ring_fluid in (false, true)

    force, exact = cylinder_force(particle_spacing; refinement, ring_fluid)
    println(rpad(particle_spacing, 9), rpad(refinement, 12),
            rpad(ring_fluid, 12), rpad(round(force.total[1], sigdigits=6), 13),
            rpad(round(force.total[2], sigdigits=3), 13),
            rpad(round(force.total[1] / exact, sigdigits=4), 11),
            round(TP.norm(force.viscous), sigdigits=3))
end
