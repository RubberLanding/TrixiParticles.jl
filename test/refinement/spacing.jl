@testset "Test `update_particle_spacing`." begin
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
                                    max_spacing_ratio=1.05,
                                    min_spacing=particle_spacing)

    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=n_layers,
                           faces=(true, true, true, false))

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
    fluid_system.cache.density .= tank.fluid.density

    semi = Semidiscretization(fluid_system, boundary_system)
    ode = semidiscretize(semi, (0, 0))

    v_ode, u_ode = ode.u0.x

    @testset "Test Uniform Particle Spacing" begin 
        particle_spacing_before = fluid_system.particle_spacing

        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode,
                                            u_ode, semi)

        particle_spacing_after = fluid_system.particle_spacing
        ref_mass_after = fluid_system.cache.reference_mass

        # Since the tank is initialized uniformly with a spacing of 0.1, 
        # and neighbors also have 0.1, the algorithm should return the exact same values.    
        @test particle_spacing_after == particle_spacing_before
        @test all(ref_mass_after .== density * particle_spacing^ndims(fluid_system))
    end 

    # Manually perturb the spacing to trigger the update logic.
    @testset "Test Smoothing Logic" begin
    center_idx = 55
    fluid_system.particle_spacing .= 0.1
    
    # IF-CASE 
    # Neighbors = 0.1, Center = 0.096
    # Ratio = 0.1 / 0.096 ≈ 1.041 < 1.1576
    # Expected: New = min(s_max, Cr * s_min)
    #               = min(0.1, 1.05 * 0.096) 
    #               = min(0.1, 0.1008) 
    #               = 0.1
    fluid_system.particle_spacing[center_idx] = 0.096
    
    TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode, semi)
    
    @test fluid_system.particle_spacing[center_idx] == 0.1
    
    # ELSE-CASE 
    # Neighbors = 0.1, Center = 0.05
    # Ratio = 0.1 / 0.05 = 2.0 > 1.1576
    # Expected: New = spacing_avg
    fluid_system.particle_spacing .= 0.1
    fluid_system.particle_spacing[center_idx] = 0.05

    # We have 28 neighbors with spacing 0.1 and the center particle with spacing 0.05
    spacing_avg = (0.05 + 28 * 0.1) / 29 

    TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode, semi)
    
    @test isapprox(fluid_system.particle_spacing[center_idx], spacing_avg) 

    # Verify mass update
    @test isapprox(fluid_system.cache.reference_mass[center_idx], density * fluid_system.particle_spacing[center_idx]^2)

    end
end