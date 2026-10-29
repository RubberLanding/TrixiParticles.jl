@testset verbose=true "`apply_refinement_criteria!`" begin
    particle_spacing = 0.1
    n_particles = 10
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257.0

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    # The wall spacing is `particle_spacing / 2 = 0.05`
    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=2, faces=(true, true, true, false),
                           spacing_ratio=2)
    wall_spacing = tank.boundary.particle_spacing

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    function setup(refinement_criteria)
        refinement = ParticleRefinement(; spacing_ratio=1.05, min_spacing=wall_spacing,
                                        max_spacing=particle_spacing,
                                        reference_density=density,
                                        smoothing_length_factor=1.5,
                                        refinement_criteria)

        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length, state_equation,
                                                   density_calculator=SummationDensity(),
                                                   particle_refinement=refinement)

        semi = Semidiscretization(fluid_system, boundary_system)
        ode = semidiscretize(semi, (0, 0))
        v_ode, u_ode = ode.u0.x

        return fluid_system, semi, v_ode, u_ode
    end

    # Fluid particles closer to the wall than the kernel support
    y = tank.fluid.coordinates[2, :]
    x = tank.fluid.coordinates[1, :]
    distance_to_wall = min.(x, width .- x, y)
    near_wall = distance_to_wall .< 0.5 * particle_spacing
    far_from_wall = distance_to_wall .> 3 * smoothing_length

    @testset "Spatial Criterion" begin
        fluid_system, semi, v_ode, u_ode = setup(SpatialRefinementCriterion())
        (; _particle_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(is_fixed[near_wall])
        @test all(_particle_spacing[near_wall] .== wall_spacing)

        @test !any(is_fixed[far_from_wall])
        @test all(_particle_spacing[far_from_wall] .== particle_spacing)

        # The spacing of the previous call is overwritten near boundaries
        # and kept everywhere else
        _particle_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(_particle_spacing[near_wall] .== wall_spacing)
        @test all(_particle_spacing[far_from_wall] .== 0.02)
    end

    @testset "Solution Criterion Finer Than Boundary" begin
        target = 0.03
        solution_criterion = SolutionRefinementCriterion((system, particle, v, u,
                                                          semi) -> target)

        # The solution criterion is listed first, but the spatial criterion is applied first
        fluid_system, semi, v_ode,
        u_ode = setup((solution_criterion, SpatialRefinementCriterion()))
        (; _particle_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # The solution criterion can decrease the boundary spacing
        @test all(_particle_spacing[near_wall] .== target)

        # All other particles take the target spacing and are fixed
        @test all(is_fixed)
        @test all(_particle_spacing[far_from_wall] .== target)

        # A finer spacing of the previous call is kept away from boundaries,
        # but overwritten by the boundary spacing near boundaries
        _particle_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)
        @test all(_particle_spacing[far_from_wall] .== 0.02)
        @test all(_particle_spacing[near_wall] .== target)
    end

    @testset "Solution Criterion Coarser Than Boundary" begin
        # Coarser than the wall spacing, but finer than `max_spacing`
        target = 0.08
        solution_criterion = SolutionRefinementCriterion((system, particle, v, u,
                                                          semi) -> target)

        fluid_system, semi, v_ode,
        u_ode = setup((SpatialRefinementCriterion(), solution_criterion))
        (; _particle_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # The solution criterion cannot increase the boundary spacing
        @test all(_particle_spacing[near_wall] .== wall_spacing)
        @test all(_particle_spacing[far_from_wall] .== target)
        @test all(is_fixed)
    end

    @testset "Inactive Solution Criterion" begin
        solution_criterion = SolutionRefinementCriterion((system, particle, v, u,
                                                          semi) -> particle_spacing)
        fluid_system, semi, v_ode, u_ode = setup(solution_criterion)
        (; _particle_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # A target spacing of `max_spacing` does not fix any particle
        @test !any(is_fixed)
        @test all(_particle_spacing .== particle_spacing)
    end
end
