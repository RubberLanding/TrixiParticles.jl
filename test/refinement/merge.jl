@testset "Particle Splitting with Resize" begin
    coordinates = [0.0 0.1 -0.12 0.5 0.8 0.9 0.0
                   0.0 0.0 0.0 0.0 0.0 0.0 0.9]

    velocity = [1.0 2.0 0.0 0.0 -1.0 -2.0 0.0
                0.0 0.0 0.0 0.0 0.0 0.0 0.1]

    density = 257.0
    NDIMS, n_particles = size(coordinates)
    mass = ones(n_particles)
    density_ = density * ones(n_particles)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 0.15
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    # Set particle spacing to init nhs correctly
    fluid = InitialCondition(; coordinates, velocity, mass, density=density_,
                             particle_spacing=smoothing_length)

    resize_buffer = ResizeBuffer(fluid)
    refinement = ParticleRefinement(n_particles=n_particles,
                                    spacing_ratio=1.05,
                                    min_spacing=1.0,
                                    resize_buffer=resize_buffer)

    density_calculator = SummationDensity()
    fluid_system = WeaklyCompressibleSPHSystem(fluid; density_calculator,
                                               state_equation,
                                               smoothing_kernel, smoothing_length,
                                               particle_refinement=refinement)
    fluid_system.cache.density .= fluid.density

    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 1.0))

    v_initial, u_initial = ode.u0.x
    v_ode = copy(v_initial)
    u_ode = copy(u_initial)

    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

    # Save original states of the particles not being merged
    orig_pos_3 = copy(u[:, 3])
    orig_vel_3 = copy(v[:, 3])
    orig_h_3 = fluid_system.smoothing_length[3]
    orig_m_3 = fluid_system.mass[3]

    orig_pos_4 = copy(u[:, 4])
    orig_vel_4 = copy(v[:, 4])
    orig_h_4 = fluid_system.smoothing_length[4]
    orig_m_4 = fluid_system.mass[4]

    orig_pos_7 = copy(u[:, 7])
    orig_vel_7 = copy(v[:, 7])
    orig_h_7 = fluid_system.smoothing_length[7]
    orig_m_7 = fluid_system.mass[7]

    # Prepare the fluid system for refinement evaluation
    TrixiParticles.reset_refinement!(fluid_system, semi)

    # Trigger the merging
    for particle in TrixiParticles.eachparticle(fluid_system)
        fluid_system.cache.reference_mass[particle] = TrixiParticles.hydrodynamic_mass(fluid_system,
                                                                                       particle)
    end
    fluid_system.cache.reference_mass[[1, 2, 3, 5, 6, 7]] .= 2.0

    # Mark particle 7 as split
    (; split_candidates) = refinement
    for particle in eachindex(split_candidates)
        split_candidates[particle] = false
    end
    split_candidates[7] = true

    # Perform two merging iterations
    # Round 1:
    #   Particle 2 should merge into particle 1
    #   Particle 3 is competing with particle 2 to merge with 1 but is ignored
    #   Particle 4 should be ignored since its mass is too high to trigger merging
    #   Particle 6 should merge into particle 5
    #   Particle 7 should be ignored since its marked as split
    # Round 2:
    #   No particles are merged. Particle 1 and 3 are close enough, but their combined mass is too high

    TrixiParticles.merge_particles!(fluid_system, refinement, v_ode, u_ode, semi,
                                    merge_iter=2)

    @testset "Test `collect_merge_candidates!`" begin
        @test refinement.delete_candidates[1] == false
        @test refinement.delete_candidates[2] == true   # Absorbed by 1 
        @test refinement.delete_candidates[3] == false  # Rejected by 1 
        @test refinement.delete_candidates[4] == false  # Untouched spectator
        @test refinement.delete_candidates[5] == false
        @test refinement.delete_candidates[6] == true   # Absorbed by 5
        @test refinement.delete_candidates[7] == false
    end

    @testset "Test `apply_merging!`" begin
        # Test that particle 2 was deleted
        @test refinement.delete_candidates[2] == true
        @test fluid_system.mass[2] == 0.0
        @test all(iszero, v[:, 2])
        @test all(isinf, u[:, 2]) # typemax(Float64) is Inf

        # Test that particle 6 was deleted
        @test refinement.delete_candidates[6] == true
        @test fluid_system.mass[6] == 0.0
        @test all(iszero, v[:, 6])
        @test all(isinf, u[:, 6])

        # Test that particle 3 did not change
        @test fluid_system.mass[3] == orig_m_3
        @test u[:, 3] == orig_pos_3
        @test v[:, 3] == orig_vel_3
        @test fluid_system.smoothing_length[3] == orig_h_3

        # Test that particle 4 did not change
        @test fluid_system.mass[4] == orig_m_4
        @test u[:, 4] == orig_pos_4
        @test v[:, 4] == orig_vel_4
        @test fluid_system.smoothing_length[4] == orig_h_4

        # Test that particle 7 did not change
        @test fluid_system.mass[7] == orig_m_7
        @test u[:, 7] == orig_pos_7
        @test v[:, 7] == orig_vel_7
        @test fluid_system.smoothing_length[7] == orig_h_7

        # Test that particle 2 got merged into 1
        m_merge = 2.0 # 1.0 + 1.0
        @test fluid_system.mass[1] ≈ m_merge

        expected_pos_1 = (1.0 * [0.0, 0.0] + 1.0 * [0.1, 0.0]) / m_merge
        @test u[:, 1] ≈ expected_pos_1

        expected_vel_1 = (1.0 * [1.0, 0.0] + 1.0 * [2.0, 0.0]) / m_merge
        @test v[:, 1] ≈ expected_vel_1

        # Test that particle 6 got merged into 5
        @test fluid_system.mass[5] ≈ m_merge

        expected_pos_5 = (1.0 * [0.8, 0.0] + 1.0 * [0.9, 0.0]) / m_merge
        @test u[:, 5] ≈ expected_pos_5

        expected_vel_5 = (1.0 * [-1.0, 0.0] + 1.0 * [-2.0, 0.0]) / m_merge
        @test v[:, 5] ≈ expected_vel_5

        @testset "Test Smoothing Length Update" begin
            W_0_1 = TrixiParticles.kernel(smoothing_kernel, 0.0, 1.0)
            tmp_m = m_merge * W_0_1

            # Test smoothing length update for particle 1
            tmp_1 = 1.0 * TrixiParticles.kernel(smoothing_kernel,
                                          norm(expected_pos_1 - coordinates[:, 1]),
                                          smoothing_length)
            tmp_2 = 1.0 * TrixiParticles.kernel(smoothing_kernel,
                                          norm(expected_pos_1 - coordinates[:, 2]),
                                          smoothing_length)
            expected_h_1 = (tmp_m / (tmp_1 + tmp_2))^(1 / NDIMS)
            @test fluid_system.smoothing_length[1] ≈ expected_h_1

            # Test smoothing length update for particle 5
            tmp_5 = 1.0 * TrixiParticles.kernel(smoothing_kernel,
                                          norm(expected_pos_5 - coordinates[:, 5]),
                                          smoothing_length)
            tmp_6 = 1.0 * TrixiParticles.kernel(smoothing_kernel,
                                          norm(expected_pos_5 - coordinates[:, 6]),
                                          smoothing_length)
            expected_h_5 = (tmp_m / (tmp_5 + tmp_6))^(1 / NDIMS)
            @test fluid_system.smoothing_length[5] ≈ expected_h_5
        end
    end
end
