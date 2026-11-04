@testset verbose=true "`update_particle_spacing`" begin
    particle_spacing = 0.1
    min_spacing = 0.05
    n_particles = 10
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257.0
    spacing_ratio = 1.05

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=1, faces=(true, true, true, false))

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    refinement = ParticleRefinement(; spacing_ratio, min_spacing,
                                    max_spacing=particle_spacing,
                                    reference_density=density,
                                    smoothing_length_factor=1.5)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                               smoothing_length, state_equation,
                                               density_calculator=SummationDensity(),
                                               particle_refinement=refinement)

    semi = Semidiscretization(fluid_system, boundary_system)
    ode = semidiscretize(semi, (0, 0))
    v_ode, u_ode = ode.u0.x

    (; _particle_spacing, reference_mass, is_fixed) = fluid_system.cache
    center = argmin(vec(sum(abs2, tank.fluid.coordinates .- [width / 2, height / 2],
                            dims=1)))
    n_fluid = nparticles(fluid_system)

    @testset "Initialization From Mass" begin
        @test all(isapprox.(_particle_spacing, particle_spacing))
        @test all(isapprox.(reference_mass, tank.fluid.mass))
    end

    @testset "Uniform Spacing" begin
        fill!(is_fixed, false)
        _particle_spacing .= particle_spacing

        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode,
                                               semi)

        # min(max_spacing, C_r * Δs) = max_spacing
        @test all(isapprox.(_particle_spacing, particle_spacing))
        @test all(isapprox.(reference_mass, density * particle_spacing^2))
    end

    @testset "Coarsening of an Unfixed Region" begin
        fill!(is_fixed, false)
        _particle_spacing .= min_spacing

        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode,
                                               semi)

        # The global `max_spacing` caps the coarsening, not the neighborhood maximum
        @test all(isapprox.(_particle_spacing, spacing_ratio * min_spacing))
    end

    @testset "Persistent Graded Band" begin
        # Fix a strip of fine particles at the left, like the particles next to a boundary
        fixed_strip = tank.fluid.coordinates[1, :] .< 0.3
        n_fixed = count(fixed_strip)

        fill!(is_fixed, false)
        _particle_spacing .= particle_spacing
        _particle_spacing[fixed_strip] .= min_spacing
        is_fixed[fixed_strip] .= true

        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode,
                                               semi)

        # The fixed particles keep their spacing
        @test all(_particle_spacing[fixed_strip] .== min_spacing)
        @test all(reference_mass[fixed_strip] .≈ density * min_spacing^2)

        n_refined_1 = count(<(particle_spacing), _particle_spacing)
        @test n_fixed < n_refined_1 < n_fluid

        # All refined particles are between the fixed spacing and `max_spacing`
        @test all(min_spacing .<= _particle_spacing .<= particle_spacing)

        # The spacing of the previous call is used, so the band grows
        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode,
                                               semi)
        n_refined_2 = count(<(particle_spacing), _particle_spacing)
        @test n_refined_2 > n_refined_1
        @test all(_particle_spacing[fixed_strip] .== min_spacing)
    end

    @testset "Geometric Mean" begin
        # Neighbors with 0.1 and the center with 0.05: ratio 2 > C_r^3,
        # so the geometric mean of the neighborhood is used.
        fill!(is_fixed, false)
        _particle_spacing .= particle_spacing
        _particle_spacing[center] = min_spacing

        _, _,
        spacing_avg = TrixiParticles.min_max_avg_spacing(fluid_system, semi, u_ode,
                                                         tank.fluid.coordinates,
                                                         center)

        TrixiParticles.update_particle_spacing(fluid_system, refinement, v_ode, u_ode,
                                               semi)

        @test _particle_spacing[center] ≈ spacing_avg
        @test min_spacing < spacing_avg < particle_spacing

        # With `n` neighbors (including the center), the geometric mean is
        # `0.1 * 0.5^(1 / n)`, which is smaller than the arithmetic mean `0.1 - 0.05 / n`
        n = round(Int,
                  log(min_spacing / particle_spacing) /
                  log(spacing_avg / particle_spacing))
        @test spacing_avg ≈ particle_spacing * (min_spacing / particle_spacing)^(1 / n)
        @test spacing_avg < particle_spacing - (particle_spacing - min_spacing) / n
    end
end
