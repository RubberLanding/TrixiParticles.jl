@inline shift_particles!(system, v_ode, u_ode, semi, integrator) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi, integrator)
    return shift_particles!(system, system.particle_refinement, v_ode, u_ode, semi, integrator)
end 

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi, integrator) = system 

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi, integrator)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    update_shifting_inner!(system, refinement, v, u, v_ode, u_ode, semi)

    apply_particle_shifting!(u_ode, v_ode, refinement, system, semi, integrator.dt)

    return system
end

function apply_particle_shifting!(u_ode, v_ode, refinement::ParticleRefinement, 
                                  system, semi, dt; theta=1.0)
    (; cache) = system
    (; delta_v, grad_density, grad_velocity) = cache

    NDIMS = ndims(system)

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    NDIMS = ndims(system)

    # Add δr from the cache to the current coordinates
    @threaded semi for particle in eachparticle(system)
        # TODO: Check if this is indeed incorrect.
        # Haftu et al. (2023) introduces a mathematical inconsistency.
        # Eq. 36 scales the position shift by theta.
        # However, Eq. 37 omits theta when updating properties via the Taylor series.
        # To maintain a mathematically consistent Taylor series expansion, the properties 
        # must be evaluated using the actual applied displacement (theta * dr_i).
        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i

            # δr_i = dt * v_shift_i
            dr_i = dt * delta_v[idx]

            # Eq 36: r'_i = r_i + θ * δr_i
            @inbounds u[i, particle] += theta * dr_i

            # Eq 37 for ρ: ρ'_i = ρ_i + ∇ρ_i ⋅ δr_i
            @inbounds current_density(v, system)[particle] += grad_density[idx] * dr_i * theta

            for j in 1:NDIMS
                idx_3d = (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i

                # Eq 37 for v: v'_i = v_i + ∇v_i ⋅ δr_i
                @inbounds v[j, particle] += grad_velocity[idx_3d] * dr_i * theta
            end
        end
    end

    return u
end

@fastpow function update_shifting_inner!(system, refinement::ParticleRefinement,
                                         v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; delta_v, grad_density, grad_velocity) = cache

    NDIMS = ndims(system)

    set_zero!(delta_v)
    set_zero!(grad_density)
    set_zero!(grad_velocity)

    v_max_ = v_max(refinement.shifting_technique, v, system)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = @inbounds current_density(v, system, particle)
            rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

            h_a = smoothing_length(system, particle)
            h_b = smoothing_length(neighbor_system, neighbor)
            h = 0.5 * (h_a + h_b)

            dx = particle_spacing(system, particle)
            Wdx = kernel(smoothing_kernel, dx, h)

            kernel_weight = kernel(smoothing_kernel, distance, h)
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h)

            # Compute density gradient 
            grad_density_ = (m_b / rho_b) * (rho_b - rho_a) * grad_kernel  

            # Compute velocity gradient 
            v_a = current_velocity(v, system, particle)
            v_b = current_velocity(v_neighbor, neighbor_system, neighbor)
            grad_velocity_ = (m_b / rho_b) * (v_b - v_a) * (grad_kernel)'

            # Eq. 7 in Sun et al. (2017). R = 0.2 and n = 4 according to p. 29 below Eq. 9.
            # According to the paper, CFL * Ma can be rewritten as Δt * v_max / h
            # (see p. 29, right above Eq. 9), but this does not yield the same amount
            # of shifting when scaling h.
            # When setting CFL * Ma = Δt * v_max / (2 * Δx), PST works as expected
            # for both small and large smoothing length factors.
            # We need to scale
            # - quadratically with the smoothing length,
            # - linearly with the particle spacing,
            # - linearly with the time step.
            # See https://github.com/trixi-framework/TrixiParticles.jl/pull/834.
            delta_v_ = -v_max_ * (2 * h)^2 / (2 * dx) * (1 + (kernel_weight / Wdx)^4 * 2 / 10) *
                       m_b / (rho_a + rho_b) * grad_kernel

            # Write into the buffers
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end

            for i in eachindex(grad_density_)
                @inbounds grad_density[i, particle] += grad_density_[i]
            end 

            for j in axes(grad_velocity_, 2)
                for i in axes(grad_velocity_, 1)
                    @inbounds grad_velocity[i, j, particle] += grad_velocity_[i, j]
                end 
            end 
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi)

    return system
end
