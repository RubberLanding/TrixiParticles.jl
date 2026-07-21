@testset "Test Resize for Systems and Semidiscretization" begin
    function create_test_system(n_particles_per_dim)
        particle_spacing = 0.1
        width = particle_spacing * n_particles_per_dim
        height = particle_spacing * n_particles_per_dim
        density = 257
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.5 * particle_spacing
        state_equation = StateEquationCole(sound_speed=10, reference_density=density, exponent=7)

        tank = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, n_layers=1, faces=(true, true, true, false))
        
        n_particles = prod(tank.n_particles_per_dimension)
        buffer = ResizeBuffer(tank.fluid)
        refinement = ParticleRefinement(n_particles=n_particles,
                                        spacing_ratio=1.05,
                                        min_spacing=particle_spacing,
                                        resize_buffer=buffer)

        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, SummationDensity(),
                                            state_equation, smoothing_kernel, smoothing_length,
                                            particle_refinement=refinement)
        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                    tank.boundary.mass,
                                                    state_equation=state_equation,
                                                    AdamiPressureExtrapolation(),
                                                    smoothing_kernel, smoothing_length)
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

        fluid_system.cache.density .= tank.fluid.density
        return fluid_system, boundary_system
    end

    fluid_system_expand, boundary_system_expand = create_test_system(11)
    fluid_system_constant, boundary_system_constant = create_test_system(12)
    fluid_system_shrink, boundary_system_shrink = create_test_system(13)

    n_particles_expand = TrixiParticles.nparticles(fluid_system_expand)
    n_particles_shrink = TrixiParticles.nparticles(fluid_system_shrink)
    n_particles_constant = TrixiParticles.nparticles(fluid_system_constant)

    u_vars = TrixiParticles.u_nvariables(fluid_system_expand)

    semi = Semidiscretization(fluid_system_expand, fluid_system_constant, fluid_system_shrink, 
                            boundary_system_expand, boundary_system_constant, boundary_system_shrink)
    ode = semidiscretize(semi, (0.0, 1.0))

    v_ode, u_ode = ode.u0.x
    v_tmp, u_tmp = similar.(ode.u0.x)    
    len_v_original, len_u_original = length(v_ode), length(u_ode)

    # Prepare shrinking system 
    refinement_shrink = fluid_system_shrink.particle_refinement
    fill!(refinement_shrink.delete_candidates, false)
    refinement_shrink.delete_candidates[[2, 4, 6, 8]] .= true
    
    refinement_shrink.resize_buffer.n_delete_particles[1] = 4
    refinement_shrink.resize_buffer.n_add_particles[1] = 1
    refinement_shrink.resize_buffer.n_new_particles[1] = n_particles_shrink - 3
    refinement_shrink.n_current_particles[1] = n_particles_shrink
    
    TrixiParticles.resize_buffer!(refinement_shrink.resize_buffer, fluid_system_shrink, 1)
    refinement_shrink.resize_buffer.masses_new .= 111.0

    # Prepare expanding system 
    refinement_expand = fluid_system_expand.particle_refinement
    fill!(refinement_expand.delete_candidates, false)
    refinement_expand.delete_candidates[[3, 7]] .= true
    
    refinement_expand.resize_buffer.n_delete_particles[1] = 2
    refinement_expand.resize_buffer.n_add_particles[1] = 5
    refinement_expand.resize_buffer.n_new_particles[1] = n_particles_expand + 3
    refinement_expand.n_current_particles[1] = n_particles_expand
    
    TrixiParticles.resize_buffer!(refinement_expand.resize_buffer, fluid_system_expand, 5)
    refinement_expand.resize_buffer.masses_new .= 222.0

    # Prepare constant system 
    refinement_constant = fluid_system_constant.particle_refinement
    fill!(refinement_constant.delete_candidates, false)
    refinement_constant.delete_candidates[[1, 5, 9]] .= true
    
    refinement_constant.resize_buffer.n_delete_particles[1] = 3
    refinement_constant.resize_buffer.n_add_particles[1] = 3
    refinement_constant.resize_buffer.n_new_particles[1] = n_particles_constant
    refinement_constant.n_current_particles[1] = n_particles_constant
    
    TrixiParticles.resize_buffer!(refinement_constant.resize_buffer, fluid_system_constant, 3)
    refinement_constant.resize_buffer.masses_new .= 333.0

    # Apply global resize
    TrixiParticles.resize!(v_ode, u_ode, v_tmp, u_tmp, semi)

    @testset "Global Memory Boundaries" begin
        # Total number of particles in the simulation should not change 
        expected_total_particles = n_particles_expand + n_particles_shrink + n_particles_constant
        @test length(u_ode) == len_u_original
        
        # Test if the ranges got updated correctly
        fluid_system_expand_expected_end = (n_particles_expand + 3) * u_vars
        fluid_system_constant_expected_end = fluid_system_expand_expected_end + (n_particles_constant * u_vars)
        fluid_system_shrink_expected_end = fluid_system_constant_expected_end + (n_particles_shrink - 3) * u_vars

        @test semi.ranges_u[1] == 1 : fluid_system_expand_expected_end
        @test semi.ranges_u[2] == (fluid_system_expand_expected_end + 1) : fluid_system_constant_expected_end
        @test semi.ranges_u[3] == (fluid_system_constant_expected_end + 1) : fluid_system_shrink_expected_end
    end

    @testset " Test Shrinking System" begin
        @test TrixiParticles.nparticles(fluid_system_shrink) == n_particles_shrink - 3
        @test fluid_system_shrink.particle_refinement.n_current_particles[1] == n_particles_shrink - 3
        
        # The single added particle should have overwritten the first deleted index (2)
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_shrink, 2), 111.0)
    end

    @testset "Test Expanding System" begin
        @test TrixiParticles.nparticles(fluid_system_expand) == n_particles_expand + 3
        @test fluid_system_expand.particle_refinement.n_current_particles[1] == n_particles_expand + 3
        
        # The first 2 added particles overwrite indices 3 and 7
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_expand, 3), 222.0)
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_expand, 7), 222.0)
        
        # The remaining 3 appended sequentially to the end
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_expand, n_particles_expand + 1), 222.0)
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_expand, n_particles_expand + 3), 222.0)
    end

    @testset "Test Constant System" begin
        @test TrixiParticles.nparticles(fluid_system_constant) == n_particles_constant
        @test fluid_system_constant.particle_refinement.n_current_particles[1] == n_particles_constant
        
        # The 3 added particles replaced the 3 deleted indices 
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_constant, 1), 333.0)
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_constant, 5), 333.0)
        @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system_constant, 9), 333.0)
    end
end