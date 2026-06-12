@testset "Test `apply_refinement_criteria`." begin
    particle_spacing = 0.1
    n_particles = 10
    n_layers = 1
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    refinement = ParticleRefinement(n_particles=n_particles,
                                    smoothing_length=smoothing_length,
                                    initial_particle_spacing=particle_spacing,
                                    spacing_ratio=1.05,
                                    min_spacing=particle_spacing)

    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=n_layers,
                           faces=(true, true, true, false), spacing_ratio=2)

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, SummationDensity(),
                                               state_equation,
                                               smoothing_kernel, smoothing_length,
                                               particle_refinement=refinement)
    particle_spacing_before = fluid_system.particle_spacing

    semi = Semidiscretization(fluid_system, boundary_system)
    ode = semidiscretize(semi, (0, 0))

    v_ode, u_ode = ode.u0.x

    TrixiParticles.set_refinement_spacing!(fluid_system, boundary_system,
                                         fluid_system.initial_condition.coordinates, v_ode,
                                         u_ode, semi)

    particle_spacing_after = fluid_system.particle_spacing

    # TODO: Test if setting the new spacing works correctly
end
