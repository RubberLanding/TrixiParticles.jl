@inline shift_particles!(system, v_ode, u_ode, semi, dt, theta, iter) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi, dt=0.01,
                                  theta=1.0, iter=3)
    return shift_particles!(system, system.particle_refinement, v_ode, u_ode, semi, dt,
                            theta, iter)
end

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi, dt, theta, iter) = system

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi, dt=0.01,
                                  theta=1.0, iter=3)
    (; delta_v) = system.cache

    NDIMS = ndims(system)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    neighborhood_search = get_neighborhood_search(system, semi)
    u_initial = copy(u)

    for i in 1:iter
        # TODO: This uses not the shifting formulation from the paper.
        # It uses the TrixiParticles shifting instead.
        update_shifting_inner!(system, refinement, v, u, v_ode, u_ode, semi) # Refinement
        apply_shift!(u, system, semi, theta) # Refinement

        # # Calculate the gradient 
        # update_shifting_inner!(system, system.shifting_technique,
        #                                         v, u, v_ode, u_ode, semi)

        # # Apply the shifting
        # for particle in eachparticle(system)
        #     for i in 1:NDIMS
        #         idx = (particle - 1) * NDIMS + i
        #         @inbounds u[i, particle] += dt * delta_v[idx]
        #     end
        # end

        # Update the NHS with the shifted positions
        update_nhs!(neighborhood_search,
                    system,
                    system, u, u,
                    semi)
    end

    # TrixiParticles.@autoinfiltrate

    # Apply property corrections once using the total displacement
    apply_property_corrections!(v, u, u_initial, system, semi)

    return system
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

            # if 0.488 < u[1, particle] < 0.489 && 0.494 < u[2, particle] < 0.495
            #     println("P: $particle | N: $neighbor | Dist: ", round(distance, digits=4), 
            #             " | Mass: ", round(m_b, digits=4), 
            #             " | dX: ", round(delta_r_[1], digits=6), 
            #             " | dY: ", round(delta_r_[2], digits=6))
            # end

            # Write into the buffers (reusing delta_v to hold delta_r)
            for i in eachindex(delta_r_)
                idx = (particle - 1) * NDIMS + i
                @inbounds delta_v[idx] += delta_r_[i]
            end

            # Compute density gradient 
            grad_density_ = (m_b / rho_b) * (rho_b - rho_a) * grad_kernel

            for i in eachindex(grad_density_)
                idx = (particle - 1) * NDIMS + i
                @inbounds grad_density[idx] += grad_density_[i]
            end

            # Compute velocity gradient 
            v_a = current_velocity(v, system, particle)
            v_b = current_velocity(v_neighbor, neighbor_system, neighbor)
            grad_velocity_ = (m_b / rho_b) * (v_b - v_a) * (grad_kernel)'

            # TODO: This just works for 2D, make it dimension-agnostic
            for j in axes(grad_velocity_, 2)
                for i in axes(grad_velocity_, 1)
                    idx_3d = (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i
                    @inbounds grad_velocity[idx_3d] += grad_velocity_[i, j]
                end
            end

            # Compute pressure gradient
            grad_pressure_ = (m_b / rho_b) * (p_b - p_a) * grad_kernel

            for i in eachindex(grad_pressure_)
                idx = (particle - 1) * NDIMS + i
                @inbounds grad_pressure[idx] += grad_pressure_[i]
            end
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi)

    return system
end

function apply_shift!(u, system, semi, theta_global=1.0)
    (; cache) = system
    (; delta_v) = cache # delta_v currently holds the spatial shift delta_r
    NDIMS = ndims(system)

    @threaded semi for particle in eachparticle(system)
        shift_norm = zero(eltype(u))
        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i
            shift_norm += delta_v[idx]^2
        end
        shift_norm = sqrt(shift_norm)

        # Dynamic theta cap (Eq. 16) based on local h_i
        h_i = smoothing_length(system, particle)
        theta = shift_norm > 0.25 * h_i ? (0.25 * h_i) / shift_norm : 1.0

        # Eq 36: Position Update
        for i in 1:NDIMS
            idx = (particle - 1) * NDIMS + i
            @inbounds u[i, particle] += theta * delta_v[idx]
        end
    end
end

function apply_property_corrections!(v, u, u_initial, system, semi)
    (; cache) = system
    (; grad_density, grad_velocity, grad_pressure) = cache
    NDIMS = ndims(system)

    @threaded semi for particle in eachparticle(system)

        # Apply Taylor series corrections
        for i in 1:NDIMS
            # Extract the total shift dr_i across all iterations
            @inbounds dr_i = u[i, particle] - u_initial[i, particle]
            idx = (particle - 1) * NDIMS + i

            # Eq 37: Taylor series correction for Density
            @inbounds current_density(v, system)[particle] += grad_density[idx] * dr_i

            # Eq 37: Taylor series correction for Velocity
            for j in 1:NDIMS
                idx_3d = (particle - 1) * NDIMS * NDIMS + (j - 1) * NDIMS + i
                @inbounds v[j, particle] += grad_velocity[idx_3d] * dr_i
            end

            # Eq 37: Taylor series correction for Pressure
            @inbounds v[NDIMS + 1, particle] += grad_pressure[idx] * dr_i
        end
    end
end

# Fallback for kernels that do not have a inflection point implemented yet
function kernel_inflection_point(::AbstractSmoothingKernel)
    error("Inflection point not defined for this kernel.")
end

@inline kernel_inflection_point(::SchoenbergQuinticSplineKernel) = 0.759298480738450
