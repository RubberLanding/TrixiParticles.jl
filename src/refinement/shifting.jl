@inline shift_particles!(system, v_ode, u_ode, semi, dt, theta) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi, dt, theta=1.0)
    return shift_particles!(system, system.particle_refinement, v_ode, u_ode, semi, dt, theta)
end

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi, dt, theta) = system

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi, dt, theta)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    update_shifting_inner!(system, refinement, v, u, v_ode, u_ode, semi)

    apply_particle_shifting!(u_ode, v_ode, refinement, system, semi, dt, theta)

    return system
end

function apply_particle_shifting!(u_ode, v_ode, refinement::ParticleRefinement,
                                  system, semi, dt, theta=1.0)
    (; cache) = system
    # NOTE: You must add `grad_pressure` to your cache allocation in refinement.jl!
    (; delta_v, grad_density, grad_velocity, grad_pressure) = cache

    NDIMS = ndims(system)

    u = wrap_u(u_ode, system, semi)
    v = wrap_v(v_ode, system, semi)

    u_before = copy(u)
    shift_ = zeros(size(u))

    # Add δr from the cache to the current coordinates
    @threaded semi for particle in eachparticle(system)

        # Calculate the norm of the shift vector (delta_r = dt * delta_v)
        shift_norm = zero(eltype(u))
        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i
            shift_norm += (delta_v[idx])^2
        end
        shift_norm = sqrt(shift_norm)

        # Dynamic theta cap based on the local particle's h_i (Haftu Eq 16)
        h_i = smoothing_length(system, particle)
        theta = shift_norm > 0.25 * h_i ? (0.25 * h_i) / shift_norm : 1.0

        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i

            # Actual applied displacement
            dr_i = theta * delta_v[idx]

            shift_[i, particle] += dr_i

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

    # shift_ = [norm(x) for x in eachcol(shift_)]

    TrixiParticles.@autoinfiltrate

    return u
end

# Fallback for kernels that might not have a mathematically useful inflection point implemented yet
kernel_inflection_point(::AbstractSmoothingKernel) = error("Inflection point not defined for this kernel.")
@inline kernel_inflection_point(::SchoenbergQuinticSplineKernel) = 0.759298480738450

@fastpow function update_shifting_inner!(system, refinement::ParticleRefinement,
                                         v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; delta_v, grad_density, grad_velocity, grad_pressure) = cache

    NDIMS = ndims(system)

    set_zero!(delta_v)
    set_zero!(grad_density)
    set_zero!(grad_velocity)
    set_zero!(grad_pressure)

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
            h_ab = 0.5 * (h_a + h_b)

            dx = particle_spacing(system, particle)

            # Eq 15 specific definitions
            xi = kernel_inflection_point(smoothing_kernel)
            
            # Kernel evaluations for the repulsive fraction
            W_rij = kernel(smoothing_kernel, distance, h_ab)
            W_dx_xi = kernel(smoothing_kernel, dx, xi * h_ab)

            # Average kernel gradient
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

            rho0 = system.particle_refinement.reference_density

            # Repulsion factor from Eq 15
            repulsion_factor = 1.0 + 0.24 * (W_rij / W_dx_xi)^4

            # Eq 15: Spatial displacement calculation (stored in delta_v buffer)
            delta_r_ = -0.5 * h_a^2 * (m_b / rho0) * repulsion_factor * grad_kernel

            # Write into the buffers (reusing delta_v to hold delta_r)
            for i in eachindex(delta_r_)
                idx = (particle - 1) * NDIMS + i
                @inbounds delta_v[idx] += delta_r_[i]
            end
            
            # Compute density gradient 
            grad_density_ = (m_b / rho_b) * (rho_b - rho_a) * grad_kernel

            # Compute velocity gradient 
            v_a = current_velocity(v, system, particle)
            v_b = current_velocity(v_neighbor, neighbor_system, neighbor)
            grad_velocity_ = (m_b / rho_b) * (v_b - v_a) * (grad_kernel)'

            # Compute pressure gradient
            grad_pressure_ = (m_b / rho_b) * (p_b - p_a) * grad_kernel

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