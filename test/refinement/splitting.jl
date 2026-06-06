using Test

@testset "Particle Splitting with Resize" begin
    particle_spacing = 0.1
    n_particles_per_dim = 10
    n_layers = 1
    width = particle_spacing * n_particles_per_dim
    height = particle_spacing * n_particles_per_dim
    density = 257

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=n_layers,
                           faces=(true, true, true, false))

    n_particles = prod(tank.n_particles_per_dimension)

    resize_buffer = ResizeBuffer(tank.fluid)
    refinement = ParticleRefinement(n_particles=n_particles,
                                    smoothing_length=smoothing_length,
                                    initial_particle_spacing=particle_spacing,
                                    max_spacing_ratio=1.05,
                                    min_spacing=particle_spacing,
                                    resize_buffer=resize_buffer)

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
    NDIMS = ndims(fluid_system)

    semi = Semidiscretization(fluid_system, boundary_system)
    ode = semidiscretize(semi, (0.0, 1.0)) 

    # Extract current arrays and wrap them for TrixiParticles
    v_ode, u_ode = ode.u0.x

    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

    # Prepare the fluid system for refinement evaluation
    TrixiParticles.reset_refinement!(fluid_system, semi)

    for particle in TrixiParticles.eachparticle(fluid_system)
        fluid_system.cache.reference_mass[particle] = TrixiParticles.hydrodynamic_mass(fluid_system, particle)
    end 

    # Force specific particles to trigger the split condition
    # by artificially reducing their reference mass below the dynamic threshold.
    test_indices = [1, 2, 3]
    n_childs_exclude_center = TrixiParticles.nchilds(fluid_system, fluid_system.particle_refinement.splitting_pattern) - 1
    n_expected_new_particles = n_childs_exclude_center * length(test_indices)
    
    particle_mass_max = refinement.max_spacing_ratio * fluid_system.cache.reference_mass[1]
    particle_mass_diff = particle_mass_max - TrixiParticles.hydrodynamic_mass(fluid_system, 1) 
    fluid_system.cache.reference_mass[test_indices] = fluid_system.cache.reference_mass[test_indices] .- (particle_mass_diff + 0.1)

    # Manual splitting and resizing
    _n_add_particles = TrixiParticles.collect_split_candidates!(fluid_system, refinement, v, u, semi)

    @testset "Test `collect_split_candidates!`" begin 
        @test n_expected_new_particles == _n_add_particles 
        @test all(refinement.split_candidates[test_indices] .== true)
        @test count(refinement.split_candidates) == length(test_indices)
    end

    fill!(refinement.resize_buffer.n_add_particles, _n_add_particles)
    TrixiParticles.resize_buffer!(refinement.resize_buffer, fluid_system, _n_add_particles)

    # Store old state
    old_masses = [TrixiParticles.hydrodynamic_mass(fluid_system, i) for i in test_indices]
    old_h = [TrixiParticles.smoothing_length(fluid_system, i) for i in test_indices]
    old_pos = [TrixiParticles.current_coords(u, fluid_system, i) for i in test_indices]
    old_vel = [TrixiParticles.current_velocity(v, fluid_system, i) for i in test_indices]
    old_rho = [TrixiParticles.current_density(v, fluid_system, i) for i in test_indices]

    TrixiParticles.apply_splitting!(fluid_system, refinement, v, u, semi)

    @testset "Test `apply_splitting!`" begin
        for (idx, particle) in enumerate(test_indices)
            expected_mass = old_masses[idx] / TrixiParticles.nchilds(fluid_system, refinement.splitting_pattern)
            expected_h = old_h[idx] * refinement.splitting_pattern.alpha
            
            @test isapprox(TrixiParticles.hydrodynamic_mass(fluid_system, particle), expected_mass)
            @test isapprox(TrixiParticles.smoothing_length(fluid_system, particle), expected_h)
            
            for child_local in 1:n_childs_exclude_center
                child_global = (idx - 1) * n_childs_exclude_center + child_local
                
                @test isapprox(refinement.resize_buffer.masses_new[child_global], expected_mass)
                @test isapprox(refinement.resize_buffer.smoothing_lengths_new[child_global], expected_h)
                @test isapprox(refinement.resize_buffer.densities_new[child_global], old_rho[idx])
                
                expected_child_vel = old_vel[idx]
                expected_child_pos = old_pos[idx] + old_h[idx] * refinement.splitting_pattern.relative_position[child_local]
                
                child_offset = (child_global - 1) * NDIMS
                for dim in 1:NDIMS
                    linear_idx = child_offset + dim
                    @test isapprox(refinement.resize_buffer.velocities_new[linear_idx], expected_child_vel[dim])
                    @test isapprox(refinement.resize_buffer.positions_new[linear_idx], expected_child_pos[dim])
                end
            end
        end
    end
end