# Interaction of boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::Union{AbstractBoundarySystem, OpenBoundarySystem},
                   neighbor_system, semi)
    # TODO Solids and moving boundaries should be considered in the continuity equation
    return dv
end

# For dummy particles with `ContinuityDensity`, solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                   neighbor_system::Union{AbstractFluidSystem,
                                          OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang}},
                   semi)
    interact!(dv, v_particle_system, u_particle_system,
              v_neighbor_system, u_neighbor_system,
              particle_system,
              neighbor_system,
              semi, get_refinement(neighbor_system))
end

# For dummy particles with `ContinuityDensity`, solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                   neighbor_system::Union{AbstractFluidSystem,
                                          OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang}},
                   semi, refinement::Nothing)
    (; boundary_model) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(particle_system)
    almostzero = sqrt(eps(h^2))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Skip neighbors with the same position because the kernel gradient is zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(particle_system) && distance < almostzero && return

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                   distance, particle)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        drho_particle = Ref(zero(rho_a))
        continuity_equation!(drho_particle, density_calculator(neighbor_system),
                             m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)

        dv[end, particle] += drho_particle[]
    end

    return dv
end

# For dummy particles with `ContinuityDensity`, solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                   neighbor_system::AbstractFluidSystem,
                   semi, refinement)
    (; boundary_model) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # The boundary has constant smoothing length
    h_w = boundary_model.smoothing_length
    almostzero = sqrt(eps(h_w^2))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_symmetric_point_neighbor(neighbor_system, system, neighbor_coords, system_coords, semi;
                                     points=each_integrated_particle(neighbor_system),
                                     parallelization_backend=SerialBackend()) do neighbor, particle,
                                                                                 pos_diff, distance

        # Guard against zero distance
        skip_zero_distance(particle_system) && distance < almostzero && return

        # Get variable fluid particle smoothing length
        h_f = smoothing_length(neighbor_system, neighbor)

        # Variable smoothing length (Eq. 19, Haftu et al.)
        h_wf = 0.5 * (h_w + h_f)

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        grad_kernel = kernel_grad_unsafe(boundary_model.smoothing_kernel, pos_diff,
                                         distance, h_wf)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        drho_particle = Ref(zero(rho_a))
        continuity_equation!(drho_particle, density_calculator(neighbor_system),
                             m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)

        dv[end, particle] += drho_particle[]
    end

    return dv
end

# This is the derivative of the density summation, which is compatible with the
# `SummationDensity` pressure acceleration.
# Energy preservation tests will fail with the other formulation.
@propagate_inbounds function continuity_equation!(drho_particle, ::SummationDensity,
                                                  m_b, rho_a, rho_b, v_a, v_b,
                                                  grad_kernel, particle)
    drho_particle[] += m_b * dot(v_a - v_b, grad_kernel)

    return drho_particle
end

# This is identical to the continuity equation of the fluid
@propagate_inbounds function continuity_equation!(drho_particle, ::ContinuityDensity,
                                                  m_b, rho_a, rho_b, v_a, v_b,
                                                  grad_kernel, particle)
    drho_particle[] += rho_a / rho_b * m_b * dot(v_a - v_b, grad_kernel)

    return drho_particle
end
