# Creates a uniform grid and manually displaces a single particle
# to create a distortion. Shifting should detect this and push the
# particle back to its original position.
@testset "Generic Shifting" begin
    particle_spacing = 0.1
    density = 1.0
    particles_per_dim = 10
    NDIMS = 2
    fluid = RectangularShape(particle_spacing, (particles_per_dim, particles_per_dim),
                             (0.0, 0.0), density=density)
    fluid.velocity .= 1.0

    # Perturb a single particle in the center of the grid.
    center_idx = particles_per_dim * Int(particles_per_dim / 2) + Int(particles_per_dim / 2)
    pos_target = fluid.coordinates[:, center_idx]

    # Push the particle Left (-) and Up (+)
    perturbation_vec = [-0.2 * particle_spacing, 0.3 * particle_spacing]
    fluid.coordinates[:, center_idx] += perturbation_vec

    particle_spacing = 0.1
    smoothing_length = 1.5 * particle_spacing
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    state_equation = StateEquationCole(sound_speed=10.0,
                                       reference_density=density,
                                       exponent=7)

    shifting_technique = ParticleShiftingTechniqueSun2017()
    density_calculator = SummationDensity()
    fluid_system = WeaklyCompressibleSPHSystem(fluid; density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length,
                                               shifting_technique=shifting_technique)
    fluid_system.cache.density .= fluid.density

    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 1.0))
    dt = 0.001

    v_ode, u_ode = ode.u0.x

    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

    # Test whether the shifting pushes the perturbed particle back to its original position.
    TrixiParticles.update_shifting_inner!(fluid_system, shifting_technique, v, u, v_ode,
                                          u_ode, semi)

    delta_v = reshape(fluid_system.cache.delta_v, ndims(fluid_system), :)
    shifting_vec = delta_v[:, center_idx]

    # Test that both vectors are pointing in opposite directions.
    @test dot(shifting_vec, perturbation_vec) < 0.0

    # Explicitly test the shifting pointing Right (+) and Down (-)
    @test shifting_vec[1] > 0.0
    @test shifting_vec[2] < 0.0
end

# Initializes a non-linear continuous field on a uniform grid. 
# Verifies that the Jacobian is approximated correctly. Manually
# shifts a single particle, applies the correction, and compares 
# the particle's mutated property array directly to the analytical
# field at the new position. 
@testset "Property Correction with `ParticleRefinement`" begin
    particle_spacing = 0.1
    density = 1.0
    particles_per_dim = 10

    # Define a non-linear continuous field over a uniform grid
    fluid = RectangularShape(particle_spacing, (particles_per_dim, particles_per_dim),
                             (0.0, 0.0), density=density)
    fluid.velocity[1, :] = sin.(fluid.coordinates[1, :])
    fluid.velocity[2, :] = cos.(fluid.coordinates[2, :])

    center_idx = particles_per_dim * Int(particles_per_dim / 2) + Int(particles_per_dim / 2)
    pos_center = fluid.coordinates[:, center_idx]

    smoothing_length = 1.5 * particle_spacing
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    state_equation = StateEquationCole(sound_speed=10.0, reference_density=density,
                                       exponent=7)

    resize_buffer = ResizeBuffer(fluid)
    refinement = ParticleRefinement(n_particles=length(fluid.mass), spacing_ratio=1.05,
                                    min_spacing=particle_spacing,
                                    resize_buffer=resize_buffer)

    density_calculator = SummationDensity()
    fluid_system = WeaklyCompressibleSPHSystem(fluid; density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length,
                                               particle_refinement=refinement)
    fluid_system.cache.density .= fluid.density

    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 1.0))
    dt = 0.001

    v_ode, u_ode = ode.u0.x
    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

    # Compare the approximated with the exact gradient
    TrixiParticles.update_shifting_inner!(fluid_system, refinement, v, u, v_ode, u_ode,
                                          semi)
    grad_reference = [[cos(pos_center[1]) 0.0]; [0.0 -sin(pos_center[2])]]
    @test all(isapprox(fluid_system.cache.grad_velocity[:, :, center_idx], grad_reference,
                       atol=5e-3))

    # Manually shift a single particle
    perturbation_vec = [-0.2 * particle_spacing, 0.2 * particle_spacing]
    pos_perturbed = fluid.coordinates[:, center_idx] + perturbation_vec

    # Only test the `center_idx` particle and inject the correct delta_v
    fluid_system.cache.delta_v[:, :] .= 0.0
    fluid_system.cache.delta_v[:, center_idx] = perturbation_vec ./ dt

    TrixiParticles.apply_particle_shifting!(u_ode, v_ode, refinement, fluid_system, semi,
                                            dt)

    # Test that the shifting correctly moved the particle 
    @test u[1, center_idx] == pos_perturbed[1]
    @test u[2, center_idx] == pos_perturbed[2]

    # Test the gradient correction 
    vel_exact = [sin(pos_perturbed[1]), cos(pos_perturbed[2])]
    error_correction = norm(v[:, center_idx] - vel_exact)
    @test error_correction < 1e-3
end
