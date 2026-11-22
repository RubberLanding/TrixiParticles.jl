@testset verbose=true "Interpolation With Variable Smoothing Lengths" begin
    particle_spacing = 0.1
    smoothing_kernel = WendlandC2Kernel{2}()

    # Linear fields, so that every particle has a different value
    pressure(pos) = pos[1] + 2 * pos[2]
    velocity(pos) = SVector(pos[1], -pos[2])
    fluid = RectangularShape(particle_spacing, (10, 10), (0.0, 0.0); density=1000.0,
                             pressure, velocity)

    function create_semi(; particle_refinement=nothing)
        system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                             smoothing_length=1.2 * particle_spacing,
                                             sound_speed=10.0,
                                             density_calculator=ContinuityDensity(),
                                             particle_refinement)
        semi = Semidiscretization(system)
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x

        return semi, semi.systems[1], v_ode, u_ode
    end

    refinement = ParticleRefinement(; spacing_ratio=1.2, min_spacing=particle_spacing / 2,
                                    max_spacing=particle_spacing,
                                    reference_density=1000.0)

    points = [0.43 0.51 0.37
              0.47 0.38 0.55]

    @testset "Default Smoothing Length" begin
        _, system, _, _ = create_semi()
        @test TrixiParticles.interpolation_smoothing_length(system) ≈
              1.2 * particle_spacing

        _, system, _, _ = create_semi(; particle_refinement=refinement)
        @test isnothing(TrixiParticles.interpolation_smoothing_length(system))
    end

    @testset "Equal Smoothing Lengths" begin
        # Without refinement and with refinement, but all particles have the same
        # smoothing length, the interpolation must be the same.
        semi, system, v_ode, u_ode = create_semi()
        expected = interpolate_points(points, semi, system, v_ode, u_ode;
                                      cut_off_bnd=false)

        semi, system, v_ode, u_ode = create_semi(; particle_refinement=refinement)
        result = interpolate_points(points, semi, system, v_ode, u_ode; cut_off_bnd=false)

        @test result.pressure ≈ expected.pressure
        @test result.velocity ≈ expected.velocity
        @test result.density ≈ expected.density
        @test result.computed_density ≈ expected.computed_density
    end

    @testset "Different Smoothing Lengths" begin
        semi, system, v_ode, u_ode = create_semi(; particle_refinement=refinement)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        v = TrixiParticles.wrap_v(v_ode, system, semi)

        # Give the particles on the right a smaller smoothing length
        for particle in TrixiParticles.eachparticle(system)
            if TrixiParticles.current_coords(u, system, particle)[1] > 0.45
                TrixiParticles.set_particle_smoothing_length!(system, particle, 0.07)
            end
        end

        result = interpolate_points(points, semi, system, v_ode, u_ode; cut_off_bnd=false)

        # Reference: Shepard interpolation with the kernel of each particle
        for point in axes(points, 2)
            point_coords = points[:, point]
            shepard = 0.0
            pressure_sum = 0.0
            velocity_sum = zeros(2)
            computed_density = 0.0

            for particle in TrixiParticles.eachparticle(system)
                distance = norm(point_coords -
                                TrixiParticles.current_coords(u, system, particle))
                h = TrixiParticles.smoothing_length(system, particle)
                W = TrixiParticles.kernel(smoothing_kernel, distance, h)
                m = TrixiParticles.hydrodynamic_mass(system, particle)
                volume = m / TrixiParticles.current_density(v, system, particle)

                shepard += volume * W
                computed_density += m * W
                pressure_sum += volume * W *
                                TrixiParticles.current_pressure(v, system, particle)
                velocity_sum .+= volume * W *
                                 TrixiParticles.current_velocity(v, system, particle)
            end

            @test result.pressure[point] ≈ pressure_sum / shepard
            @test result.velocity[:, point] ≈ velocity_sum / shepard
            @test result.computed_density[point] ≈ computed_density
        end

        # A single smoothing length gives a different result
        result_fixed = interpolate_points(points, semi, system, v_ode, u_ode;
                                          smoothing_length=1.2 * particle_spacing,
                                          cut_off_bnd=false)
        @test !isapprox(result_fixed.pressure, result.pressure)
    end
end
