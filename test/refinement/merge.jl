@testset "Particle Splitting with Resize" begin
    # Particle 1 should merge with 2
    # Particle 2 should merge into particle 1
    # Particle 3 is competing with particle 2 to merge with 1 but is ignored
    # Particle 4 should be ignored since the reference mass is too low to trigger merging
    # Particle 5 should merge with 6
    # Particle 6 should merge into particle 5
    # Particle 7 should be ignored since its marked for deletion 
    # Particle 8 should be ignored since its marked as split
    
    particle_spacing = 0.1
    coordinates = [0.0  0.1 -0.12  0.5  0.8  0.9  0.85 0.0
                   0.0  0.0   0.0  0.0  0.0  0.0  0.0  0.9]
    
    # Distinct velocities to test momentum conservation independently
    velocity = [1.0  2.0  0.0  0.0 -1.0 -2.0 0.0 0.0
                0.0  0.0  0.0  0.0  0.0  0.0 0.1 0.2]
    
    density = 257.0

    NDIMS, n_particles = size(coordinates)
    mass = ones(n_particles)
    density_ = density * ones(n_particles)
    fluid = InitialCondition(; coordinates, velocity, mass, density=density_)

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    resize_buffer = ResizeBuffer(fluid)
    refinement = ParticleRefinement(n_particles=n_particles,
                                    smoothing_length=smoothing_length,
                                    initial_particle_spacing=particle_spacing,
                                    max_spacing_ratio=1.05,
                                    min_spacing=particle_spacing,
                                    resize_buffer=resize_buffer)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
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
    orig_h_3   = fluid_system.smoothing_length[3]
    orig_m_3   = fluid_system.mass[3]

    orig_pos_4 = copy(u[:, 4])
    orig_vel_4 = copy(v[:, 4])
    orig_h_4   = fluid_system.smoothing_length[4]
    orig_m_4   = fluid_system.mass[4]

    orig_pos_7 = copy(u[:, 7])
    orig_vel_7 = copy(v[:, 7])
    orig_h_7   = fluid_system.smoothing_length[7]
    orig_m_7   = fluid_system.mass[7]

    orig_pos_8 = copy(u[:, 8])
    orig_vel_8 = copy(v[:, 8])
    orig_h_8   = fluid_system.smoothing_length[8]
    orig_m_8   = fluid_system.mass[8]

    # Prepare the fluid system for refinement evaluation
    TrixiParticles.reset_refinement!(fluid_system, semi)

    for particle in TrixiParticles.eachparticle(fluid_system)
        fluid_system.cache.reference_mass[particle] = TrixiParticles.hydrodynamic_mass(fluid_system, particle)
    end 

    # Trigger the merging
    fluid_system.cache.reference_mass[[1, 2, 3, 5, 6, 7, 8]] .= 3.0

    # Mark particle 7 for deletion 
    (; delete_candidates) = refinement
    for particle in eachindex(delete_candidates)
        delete_candidates[particle] = false
    end
    delete_candidates[7] = true

    # Mark particle 8 as split
    (; split_candidates) = refinement
    for particle in eachindex(split_candidates)
        split_candidates[particle] = false
    end
    split_candidates[8] = true

    # Execute merging
    merge_particles!(semi, v_ode, u_ode)

    @testset "Test `collect_merge_candidates!`" begin
        @test refinement.delete_candidates[1] == false
        @test refinement.delete_candidates[2] == true   # Absorbed by 1 
        @test refinement.delete_candidates[3] == false  # Rejected by 1 
        @test refinement.delete_candidates[4] == false  # Untouched spectator
        @test refinement.delete_candidates[5] == false
        @test refinement.delete_candidates[6] == true   # Absorbed by 5
        @test refinement.delete_candidates[7] == true   # Already marked as deleted
        @test refinement.delete_candidates[8] == false  # Marked for split
    end

    @testset "Test `apply_merging!`" begin
        # Test that particle 2 was deleted
        @test refinement.delete_candidates[2] == true
        @test fluid_system.mass[2] == 0.0
        @test all(v[:, 2] .== 0.0)
        @test all(isinf.(u[:, 2])) # typemax(Float64) is Inf

        # Test that particle 6 was deleted
        @test refinement.delete_candidates[6] == true
        @test fluid_system.mass[6] == 0.0
        @test all(v[:, 6] .== 0.0)
        @test all(isinf.(u[:, 6]))

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

        # Test that particle 8 did not change
        @test fluid_system.mass[8] == orig_m_8
        @test u[:, 8] == orig_pos_8
        @test v[:, 8] == orig_vel_8
        @test fluid_system.smoothing_length[8] == orig_h_8

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
            tmp_1 = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_1 - coordinates[:,1]), smoothing_length)
            tmp_2 = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_1 - coordinates[:,2]), smoothing_length)
            expected_h_1 = (tmp_m / (tmp_1 + tmp_2))^(1 / NDIMS)
            @test fluid_system.smoothing_length[1] ≈ expected_h_1

            # Test smoothing length update for particle 5
            tmp_5 = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_5 - coordinates[:,5]), smoothing_length)
            tmp_6 = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_5 - coordinates[:,6]), smoothing_length)
            expected_h_5 = (tmp_m / (tmp_5 + tmp_6))^(1 / NDIMS)
            @test fluid_system.smoothing_length[5] ≈ expected_h_5
        end
    end
end