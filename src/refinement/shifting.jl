@inline shift_particles!(system, v_ode, u_ode, semi, dt) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi, dt)
    return shift_particles!(system, system.particle_refinement, v_ode, u_ode, semi, dt)
end

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi, dt) = system

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi, dt)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    update_shifting_inner!(system, refinement, v, u, v_ode, u_ode, semi)

    apply_particle_shifting!(u_ode, v_ode, refinement, system, semi, dt)

    return system
end

function apply_particle_shifting!(u_ode, v_ode, refinement::ParticleRefinement,
                                  system, semi, dt)
    (; cache) = system
    # NOTE: You must add `grad_pressure` to your cache allocation in refinement.jl!
    (; delta_v, grad_density, grad_velocity, grad_pressure) = cache

    NDIMS = ndims(system)

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    # Add δr from the cache to the current coordinates
    @threaded semi for particle in eachparticle(system)
        
        # Calculate the norm of the shift vector (delta_r = dt * delta_v)
        shift_norm = zero(eltype(u))
        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i
            shift_norm += (dt * delta_v[idx])^2
        end
        shift_norm = sqrt(shift_norm)

        # Dynamic theta cap based on the local particle's h_i (Haftu Eq 16)
        h_i = smoothing_length(system, particle)
        theta = shift_norm > 0.25 * h_i ? (0.25 * h_i) / shift_norm : 1.0

        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i

            # Actual applied displacement
            dr_i = theta * dt * delta_v[idx]

            # Eq 36: Position Update
            @inbounds u[i, particle] += dr_i

            # Eq 37: Taylor series correction for Density
            @inbounds current_density(v, system)[particle] += grad_density[idx] * dr_i

            # Eq 37: Taylor series correction for Velocity
            for j in 1:NDIMS
                idx_3d = (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i
                @inbounds v[j, particle] += grad_velocity[idx_3d] * dr_i 
            end
            
            # Eq 37: Taylor series correction for Pressure (for EDAC specifically)
            @inbounds v[NDIMS + 1, particle] += grad_pressure[idx] * dr_i
        end
    end

    return u
end

@fastpow function update_shifting_inner!(system, refinement::ParticleRefinement,
                                         v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; delta_v, grad_density, grad_velocity, grad_pressure) = cache

    NDIMS = ndims(system)

    set_zero!(delta_v)
    set_zero!(grad_density)
    set_zero!(grad_velocity)
    set_zero!(grad_pressure)

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
            
            # Extract pressure to calculate the gradient
            p_a = @inbounds current_pressure(v, system, particle)
            p_b = @inbounds current_pressure(v_neighbor, neighbor_system, neighbor)

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
            
            # Compute pressure gradient
            grad_pressure_ = (m_b / rho_b) * (p_b - p_a) * grad_kernel

            # Eq. 7 in Sun et al. (2017) adapted with Haftu multiplier
            delta_v_ = -v_max_ * (2 * h)^2 / (2 * dx) *
                       (1 + (kernel_weight / Wdx)^4 * 2 / 10) *
                       m_b / (rho_a + rho_b) * grad_kernel

            # Write into the buffers
            for i in eachindex(delta_v_)
                idx = (particle - 1) * NDIMS + i
                @inbounds delta_v[idx] += delta_v_[i]
            end

            for i in eachindex(grad_density_)
                idx = (particle - 1) * NDIMS + i
                @inbounds grad_density[idx] += grad_density_[i]
            end
            
            for i in eachindex(grad_pressure_)
                idx = (particle - 1) * NDIMS + i
                @inbounds grad_pressure[idx] += grad_pressure_[i]
            end

            # TODO: This just works for 2D, make it dimension-agnostic
            for j in axes(grad_velocity_, 2)
                for i in axes(grad_velocity_, 1)
                    idx_3d = (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i
                    @inbounds grad_velocity[idx_3d] += grad_velocity_[i, j]
                end
            end
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi)

    return system
end