# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system, semi)
    (; sound_speed, density_calculator, correction, nu_edac, particle_refinement) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(particle_system)
    almostzero = sqrt(eps(h^2))

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # Skip neighbors with the same position because the kernel gradient is zero.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(particle_system) && distance < almostzero && return

        # Now that we know that `distance` is not zero, we can safely call the unsafe
        # version of the kernel gradient to avoid redundant zero checks.
        grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                   distance, particle)

        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # This technique by Basa et al. 2017 (10.1002/fld.1927) aims to reduce numerical
        # errors due to large pressures by subtracting the average pressure of neighboring
        # particles.
        # It results in significant improvement for EDAC, especially with TVF,
        # but not for WCSPH, according to Ramachandran & Puri (2019), Section 3.2.
        # Note that the return value is zero when not using average pressure reduction.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        # Formulation by Haftu et al. adapts EDAC equations for variable smoothing lengths
        h_a = smoothing_length(particle_system, particle)
        h_b = smoothing_length(neighbor_system, neighbor)

        # TODO: Look how do extract the kernel grad the most performant way
        grad_kernel_a = kernel_grad(smoothing_kernel, pos_diff, distance, h_a)
        grad_kernel_b = kernel_grad(smoothing_kernel, pos_diff, distance, h_b)

        beta_a = get_beta(particle_system, particle, particle_system.particle_refinement)
        beta_a = abs(beta_a) < eps() ? 1.0 : beta_a
        beta_a_inv = 1.0 / beta_a

        beta_b = get_beta(neighbor_system, neighbor, neighbor_system.particle_refinement)
        beta_b = abs(beta_b) < eps() ? 1.0 : beta_b

        dv_pressure, P_a,
        P_b = evaluate_pressure_terms(particle_system, neighbor_system,
                                      particle, neighbor, m_a, m_b,
                                      p_a, p_b, rho_a, rho_b, pos_diff, distance,
                                      grad_kernel_a, grad_kernel_b,
                                      beta_a, beta_b,
                                      particle_system.particle_refinement)

        dv_particle = Ref(dv_pressure)
        @inbounds dv_shifting!(dv_particle, particle_refinement,
                               shifting_technique(particle_system),
                               particle_system, neighbor_system,
                               v_particle_system, v_neighbor_system,
                               particle, neighbor, m_a, m_b, rho_a, rho_b, v_a, v_b,
                               pos_diff, distance,
                               grad_kernel_a, grad_kernel_b, beta_a, beta_b, correction)

        grad_kernel_avg = (grad_kernel_a + grad_kernel_b) / 2
        @inbounds dv_viscosity!(dv_particle, particle_refinement, particle_system,
                                neighbor_system,
                                v_particle_system, v_neighbor_system,
                                particle, neighbor, pos_diff, distance,
                                sound_speed, m_a, m_b, rho_a, rho_b,
                                v_a, v_b, grad_kernel_a, grad_kernel_avg, beta_a_inv)

        @inbounds adhesion_force!(dv_particle, surface_tension_a, particle_system,
                                  neighbor_system,
                                  particle, neighbor, pos_diff, distance)

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[][i]
        end

        u_shift_a = delta_v(particle_system, particle)

        v_diff = v_a - v_b

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, nu_edac, P_a,
                            P_b,
                            grad_kernel_a, grad_kernel_b, beta_a_inv, u_shift_a,
                            particle_refinement)
        drho_particle = Ref(zero(rho_a))

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        @inbounds continuity_equation!(drho_particle, density_calculator,
                                       particle_system, neighbor_system,
                                       particle, neighbor, pos_diff, distance,
                                       m_b, rho_a, rho_b, v_a, v_b, grad_kernel)

        @inbounds write_drho_particle!(dv, density_calculator, drho_particle, particle)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac,
                                     P_a, P_b, grad_kernel_a, grad_kernel_b, beta_a_inv,
                                     u_shift_a, refinement)
    volume_b = m_b / rho_b

    h_a = smoothing_length(particle_system, particle)
    h_b = smoothing_length(neighbor_system, neighbor)

    # Extract the base coefficient: nu_edac = (alpha * c_s * h_ref) / 8
    h_ref = initial_smoothing_length(particle_system)
    nu_coeff = nu_edac / h_ref

    # Calculate individual EDAC viscosity (Eq. 11)
    nu_a = nu_coeff * h_a
    nu_b = nu_coeff * h_b

    pressure_diff = p_a - p_b

    # According to Haftu, this should be:
    # artificial_eos = (rho_0 / beta_a) * (m_b / rho_b) * sound_speed^2 * dot(v_diff, grad_kernel_a)
    # TODO: Do we need the reference density rho_0 or is rho_a fine?
    # Compute equation-of-state term (Eq. 6, Term 1)
    artificial_eos = rho_a * sound_speed^2 * beta_a_inv * volume_b *
                     dot(v_diff, grad_kernel_a)

    grad_kernel_avg = (grad_kernel_a + grad_kernel_b) / 2
    nu_avg = 4 * (nu_a * nu_b) / (nu_a + nu_b)
    smoothing_length_average = (h_a + h_b) / 2

    # TODO: This is not mentioned in the Haft et al. paper but should be standard practice?
    tmp = 1 / (distance^2 + smoothing_length_average^2 / 100)

    # Compute damping_term (Eq. 6, Term 2)
    damping_term = beta_a_inv * volume_b * nu_avg * pressure_diff *
                   dot(grad_kernel_avg, pos_diff) * tmp

    # Compute shifting correction term (Eq. 6, Term 3)
    shifting_correction = m_b *
                          dot(u_shift_a, (P_a .* grad_kernel_a) + (P_b .* grad_kernel_b))

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1,
       particle] += artificial_eos + damping_term + shifting_correction

    return dv
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac,
                                     P_a, P_b, grad_kernel_a, grad_kernel_b, beta_a_inv,
                                     u_shift_a, ::Nothing)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel_a)

    eta_a = rho_a * nu_edac
    eta_b = rho_b * nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    smoothing_length_average = (smoothing_length(particle_system, particle) +
                                smoothing_length(neighbor_system, neighbor)) / 2
    tmp = eta_tilde / (distance^2 + smoothing_length_average^2 / 100)

    # This formulation was introduced by Hu and Adams (2006). https://doi.org/10.1016/j.jcp.2005.09.001
    # They argued that the formulation is more flexible because of the possibility to formulate
    # different inter-particle averages or to assume different inter-particle distributions.
    # Ramachandran (2019) and Adami (2012) use this formulation also for the pressure acceleration.
    #
    # TODO: Is there a better formulation to discretize the Laplace operator?
    # Because when using this formulation for the pressure acceleration, it is not
    # energy conserving.
    # See issue: https://github.com/trixi-framework/TrixiParticles.jl/issues/394
    #
    # This is similar to density diffusion in WCSPH
    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel_a, pos_diff)

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1, particle] += artificial_eos + damping_term

    return dv
end

@inline function evaluate_pressure_terms(particle_system, neighbor_system, particle,
                                         neighbor,
                                         m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                         distance,
                                         grad_kernel_a, grad_kernel_b, beta_a, beta_b,
                                         refinement::Nothing)

    # Delegate to standard TrixiParticles architecture
    correction = system_correction(particle_system)
    dv_pressure = pressure_acceleration(particle_system, neighbor_system, particle,
                                        neighbor,
                                        m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                        distance,
                                        grad_kernel_a, correction)

    # Compute unscaled P_a and P_b for pressure_evolution
    p_avg_a = average_pressure(particle_system, particle)
    p_avg_b = average_pressure(neighbor_system, neighbor)

    P_a = (p_a - p_avg_a) / rho_a^2
    P_b = (p_b - p_avg_b) / rho_b^2

    return dv_pressure, P_a, P_b
end

@inline function evaluate_pressure_terms(particle_system, neighbor_system, particle,
                                         neighbor,
                                         m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                         distance,
                                         grad_kernel_a, grad_kernel_b, beta_a, beta_b,
                                         refinement)

    # Compute scaled P_a and P_b (Eq. 8)
    p_avg_a = average_pressure(particle_system, particle)
    p_avg_b = average_pressure(neighbor_system, neighbor)

    P_a = (p_a - p_avg_a) / (rho_a^2 * beta_a)
    P_b = (p_b - p_avg_b) / (rho_b^2 * beta_b)

    # Compute explicit physical pressure
    term_a_pressure = P_a .* grad_kernel_a
    term_b_pressure = P_b .* grad_kernel_b
    dv_pressure = -m_b * (term_a_pressure + term_b_pressure)

    return dv_pressure, P_a, P_b
end
