@testset "Empiric Error Estimation" begin
    particle_spacing = 0.1
    density = 1.0 
    particles_per_dim = 10
    fluid_baseline = RectangularShape(particle_spacing, (particles_per_dim, particles_per_dim), (0.0, 0.0), density=density)
    fluid_perturbed = RectangularShape(particle_spacing, (particles_per_dim, particles_per_dim), (0.0, 0.0), density=density)

    center_idx = particles_per_dim * Int(particles_per_dim / 2) + Int(particles_per_dim / 2)
    fluid_perturbed.coordinates[1, center_idx] += 0.2 * particle_spacing
    fluid_perturbed.coordinates[2, center_idx] += 0.2 * particle_spacing

    function measure_error(fluid, idx)
        (; coordinates) = fluid
        n_particles = nparticles(fluid)

        field = similar(fluid.coordinates)
        field[1, :] = fluid.coordinates[1,:]
        field[2, :] = fluid.coordinates[2,:]

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.5 * particle_spacing 
        state_equation = StateEquationCole(sound_speed=10.0,
                                        reference_density=density,
                                        exponent=7)

        fluid_system = WeaklyCompressibleSPHSystem(fluid,
                                            SummationDensity(),
                                            state_equation,
                                            smoothing_kernel,
                                            smoothing_length)
        semi = Semidiscretization(fluid_system)
        _ = semidiscretize(semi, (0.0, 1.0)) 

        kernel_correction_coefficient = zeros(n_particles)
        interpolated_field = zeros(size(field))

        TrixiParticles.foreach_point_neighbor(fluid_system, fluid_system, coordinates, coordinates,
                            semi) do particle, neighbor, pos_diff, distance
            rho_b = density
            m_b = fluid_system.mass[neighbor]
            volume = m_b / rho_b
            kernel_weight = TrixiParticles.kernel(smoothing_kernel, distance, smoothing_length)
            
            kernel_correction_coefficient[particle] += volume * kernel_weight
            
            interpolated_field[1, particle] += volume * kernel_weight * field[1, neighbor]
            interpolated_field[2, particle] += volume * kernel_weight * field[2, neighbor]
        end

        for particle in TrixiParticles.eachparticle(fluid_system)
            # Assume kernel coefficients to be non-zero
            interpolated_field[1, particle] /= kernel_correction_coefficient[particle]
            interpolated_field[2, particle] /= kernel_correction_coefficient[particle]
        end

        error = norm(field[:, idx] .- interpolated_field[:, idx])

        return error
    end

    error_baseline = println("Baseline Error: ", measure_error(fluid_baseline, center_idx))
    error_perturbed = println("Perturbed Error: ", measure_error(fluid_perturbed, center_idx))
end 
