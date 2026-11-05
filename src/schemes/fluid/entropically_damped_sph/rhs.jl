# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system, semi)
    (; sound_speed, density_calculator, correction, nu_edac,
     particle_refinement) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    h = initial_smoothing_length(particle_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_symmetric_point_neighbor(particle_system, neighbor_system,
                                     system_coords, neighbor_coords, semi;
                                     points=each_integrated_particle(particle_system)) do particle,
                                                                                          neighbor,
                                                                                          pos_diff,
                                                                                          distance
        # Skip neighbors with (almost) the same position because the kernel gradient
        # is zero, but computing it would divide by zero (see `almostzero`).
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(particle_system) && distance < almostzero(h) && return

        grad_kernel_a,
        grad_kernel_b = kernel_grads_ab(particle_system, neighbor_system,
                                        particle_refinement, pos_diff,
                                        distance, particle, neighbor)

        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        # Propagate `@inbounds` to the momentum equation, which accesses particle data
        dv_particle = @inbounds add_momentum_equation(zero(v_a), particle_system,
                                                      neighbor_system,
                                                      v_particle_system, v_neighbor_system,
                                                      particle, neighbor,
                                                      pos_diff, distance, grad_kernel_a,
                                                      grad_kernel_b, sound_speed, m_a, m_b,
                                                      p_a, p_b, rho_a, rho_b, v_a, v_b,
                                                      particle_refinement)

        # Extra terms in the momentum equation when using a shifting technique.
        # These are not included in `add_momentum_equation` because they must not be
        # applied to structures (see `interact_structure_fluid!`).
        dv_particle = @inbounds add_dv_shifting(dv_particle,
                                                shifting_technique(particle_system),
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, m_a, m_b, rho_a, rho_b,
                                                v_a, v_b,
                                                pos_diff, distance, grad_kernel_a,
                                                correction)

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        v_diff = v_a - v_b

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, nu_edac,
                            grad_kernel_a, grad_kernel_b, particle_refinement)

        drho_particle = zero(rho_a)

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        drho_particle = @inbounds add_continuity_equation(drho_particle, density_calculator,
                                                          particle_system, neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance, m_b, rho_a, rho_b,
                                                          v_a, v_b, grad_kernel_a)

        @inbounds write_drho_particle!(dv, density_calculator, drho_particle, particle)
    end

    return dv
end

# Add the acceleration of `particle` due to `neighbor` to `dv_particle`.
# `particle` must be in `particle_system` and `neighbor` must be in `neighbor_system`.
# This includes pressure, viscosity, surface tension and adhesion, but not the extra terms
# from shifting techniques (see `interact_structure_fluid!`).
# Note that this function is also used for the structure-fluid interaction to compute
# the exact opposite pair force. When adding new terms here, make sure that they are
# also valid for structure neighbors.
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::EntropicallyDampedSPHSystem,
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel, sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b)
    (; correction) = particle_system

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # This technique by Basa et al. 2017 (10.1002/fld.1927) aims to reduce numerical
    # errors due to large pressures by subtracting the average pressure of neighboring
    # particles.
    # It results in significant improvement for EDAC, especially with TVF,
    # but not for WCSPH, according to Ramachandran & Puri (2019), Section 3.2.
    # Note that the return value is zero when not using average pressure reduction.
    p_avg = average_pressure(particle_system, particle)

    dv_particle += pressure_acceleration(particle_system, neighbor_system,
                                         particle, neighbor,
                                         m_a, m_b, p_a - p_avg, p_b - p_avg, rho_a,
                                         rho_b, pos_diff, distance, grad_kernel,
                                         correction)

    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance,
                                   sound_speed, m_a, m_b, rho_a, rho_b,
                                   v_a, v_b, grad_kernel)

    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system,
                                         particle, neighbor, pos_diff, distance,
                                         rho_a, rho_b, grad_kernel, 1)

    dv_particle = add_dv_adhesion(dv_particle, surface_tension_a,
                                  particle_system, neighbor_system,
                                  particle, neighbor, pos_diff, distance)

    return dv_particle
end

# Without refinement, both kernel gradients are identical
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::EntropicallyDampedSPHSystem,
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel_a, grad_kernel_b,
                                                   sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b,
                                                   ::Nothing)
    return add_momentum_equation(dv_particle, particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance, grad_kernel_a,
                                 sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, v_a, v_b)
end

# With refinement, each particle uses its own smoothing length and pressures are scaled
# by `beta` (Eq. 8, Haftu et al.).
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::EntropicallyDampedSPHSystem,
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel_a, grad_kernel_b,
                                                   sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b,
                                                   refinement)
    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    beta_a = get_beta(particle_system, particle, refinement)
    beta_b = get_beta(neighbor_system, neighbor, get_refinement(neighbor_system))

    P_a,
    P_b = scaled_pressures(particle_system, particle, p_a, p_b, rho_a, rho_b,
                           beta_a, beta_b)

    dv_particle += -m_b * (P_a * grad_kernel_a + P_b * grad_kernel_b)

    grad_kernel_avg = (grad_kernel_a + grad_kernel_b) / 2
    dv_particle = add_dv_viscosity(dv_particle, refinement,
                                   particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance,
                                   sound_speed, m_a, m_b, rho_a, rho_b,
                                   v_a, v_b, grad_kernel_a, grad_kernel_avg, inv(beta_a))

    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system,
                                         particle, neighbor, pos_diff, distance,
                                         rho_a, rho_b, grad_kernel_a, 1)

    dv_particle = add_dv_adhesion(dv_particle, surface_tension_a,
                                  particle_system, neighbor_system,
                                  particle, neighbor, pos_diff, distance)

    return dv_particle
end

# Compute the scaled pressures `P_a` and `P_b` (Eq. 8, Haftu et al.)
@propagate_inbounds function scaled_pressures(particle_system, particle, p_a, p_b,
                                              rho_a, rho_b, beta_a, beta_b)
    # Shift both pressures by the same constant a
    # per-particle average-pressure would cancel the pressure gradient.
    # The source code implements it correctly, the paper not.
    p_avg = average_pressure(particle_system, particle)

    P_a = (p_a - p_avg) / (rho_a^2 * beta_a)
    P_b = (p_b - p_avg) / (rho_b^2 * beta_b)

    return P_a, P_b
end

# Without refinement, both kernel gradients are identical
@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac,
                                     grad_kernel_a, grad_kernel_b, ::Nothing)
    return pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel_a,
                               particle, neighbor, pos_diff, distance,
                               sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, nu_edac)
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac,
                                     grad_kernel_a, grad_kernel_b, refinement)
    (; reference_density) = refinement

    beta_a = get_beta(particle_system, particle, refinement)
    beta_b = get_beta(neighbor_system, neighbor, get_refinement(neighbor_system))
    beta_a_inv = inv(beta_a)

    P_a,
    P_b = scaled_pressures(particle_system, particle, p_a, p_b, rho_a, rho_b,
                           beta_a, beta_b)

    u_shift_a = delta_v(particle_system, particle)

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
    # TODO: Haftu uses a reference density \rho_0 here. I think this
    # might be the initial fluid density. Maybe we need store that in
    # the refinement struct then.  
    # Compute equation-of-state term (Eq. 6, Term 1)
    artificial_eos = reference_density * sound_speed^2 * beta_a_inv * volume_b *
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
                                     grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

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
    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1, particle] += artificial_eos + damping_term

    return dv
end
