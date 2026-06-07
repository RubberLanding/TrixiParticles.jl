@testset "Particle Merging" begin
    # Particle 1 should merge with 2 (instead of 3 which is competing) and particle 5 with 6.
    particle_spacing = 0.1
    coordinates = [0.0  0.1 -0.12  0.5  0.8  0.9
                   0.0  0.0   0.0  0.0  0.0  0.0]
    
    # Distinct velocities to test momentum conservation independently
    velocity = [1.0  2.0  0.0  0.0 -1.0 -2.0
                0.0  0.0  0.0  0.0  0.0  0.0]
    
    density = 257.0

    NDIMS, n_particles = size(coordinates)
    mass = ones(n_particles)
    density_ = density * ones(n_particles)
    fluid = InitialCondition(; coordinates, velocity, mass, density=density_)

    smoothing_kernel = SchoenbergCubicSplineKernel{NDIMS}()
    smoothing_length = 0.15
    state_equation = StateEquationCole(sound_speed=10, reference_density=density, exponent=7)

    refinement = ParticleRefinement(n_particles=n_particles,
                                    smoothing_length=smoothing_length,
                                    initial_particle_spacing=particle_spacing,
                                    max_spacing_ratio=1.05,
                                    min_spacing=particle_spacing)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
                                               state_equation, smoothing_kernel, 
                                               smoothing_length, particle_refinement=refinement)
    fluid_system.cache.density .= density

    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 0.0))

    v_ode, u_ode = ode.u0.x
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)

    # Save original states of the particles not being merged
    orig_pos_3 = copy(u[:, 3])
    orig_vel_3 = copy(v[:, 3])
    orig_h_3   = fluid_system.smoothing_length[3]
    orig_m_3   = fluid_system.mass[3]

    orig_pos_4 = copy(u[:, 4])
    orig_vel_4 = copy(v[:, 4])
    orig_h_4   = fluid_system.smoothing_length[4]
    orig_m_4   = fluid_system.mass[4]

    # Trigger merging
    fluid_system.cache.reference_mass[[1, 2, 3, 5, 6]] .= 3.0

    # Execute inner merge
    TrixiParticles.merge_particles_inner!(fluid_system, refinement, semi, v, u)

    @testset "Deletion Flags & Targeting" begin
        @test refinement.delete_candidates[1] == false
        @test refinement.delete_candidates[2] == true   # Absorbed by 1 
        @test refinement.delete_candidates[3] == false  # Rejected by 1 
        @test refinement.delete_candidates[4] == false  # Untouched spectator
        @test refinement.delete_candidates[5] == false
        @test refinement.delete_candidates[6] == true   # Absorbed by 5
    end

    @testset "Competitor & Spectator Immutability" begin
        # Competitor (Particle 3)
        @test fluid_system.mass[3] == orig_m_3
        @test u[:, 3] == orig_pos_3
        @test v[:, 3] == orig_vel_3
        @test fluid_system.smoothing_length[3] == orig_h_3

        # Spectator (Particle 4)
        @test fluid_system.mass[4] == orig_m_4
        @test u[:, 4] == orig_pos_4
        @test v[:, 4] == orig_vel_4
        @test fluid_system.smoothing_length[4] == orig_h_4
    end

    @testset "Pair A (1 & 2) Conservation & Equations" begin
        m_merge = 2.0 # 1.0 + 1.0
        @test fluid_system.mass[1] ≈ m_merge

        expected_pos_1 = (1.0 * [0.0, 0.0] + 1.0 * [0.1, 0.0]) / m_merge
        @test u[:, 1] ≈ expected_pos_1

        expected_vel_1 = (1.0 * [1.0, 0.0] + 1.0 * [2.0, 0.0]) / m_merge
        @test v[:, 1] ≈ expected_vel_1

        # Equation 34 Smoothing Length Check
        W_0_1 = TrixiParticles.kernel(smoothing_kernel, 0.0, 1.0)
        tmp_m = m_merge * W_0_1
        tmp_a = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_1 - [0.0, 0.0]), smoothing_length)
        tmp_b = 1.0 * TrixiParticles.kernel(smoothing_kernel, norm(expected_pos_1 - [0.1, 0.0]), smoothing_length)
        expected_h_1 = (tmp_m / (tmp_a + tmp_b))^(1 / NDIMS)
        
        @test fluid_system.smoothing_length[1] ≈ expected_h_1
    end

    @testset "Pair B (5 & 6) Conservation & Equations" begin
        m_merge = 2.0
        @test fluid_system.mass[5] ≈ m_merge

        expected_pos_5 = (1.0 * [0.8, 0.0] + 1.0 * [0.9, 0.0]) / m_merge
        @test u[:, 5] ≈ expected_pos_5

        expected_vel_5 = (1.0 * [-1.0, 0.0] + 1.0 * [-2.0, 0.0]) / m_merge
        @test v[:, 5] ≈ expected_vel_5
    end
end