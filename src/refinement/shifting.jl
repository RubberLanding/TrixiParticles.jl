@inline shift_particles!(system, v_ode, u_ode, semi; kwargs...) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi;
                                  kwargs...)
    return shift_particles!(system, system.particle_refinement, v_ode, u_ode, semi;
                            kwargs...)
end

@inline shift_particles!(system::AbstractFluidSystem, ::Nothing, v_ode, u_ode, semi;
                         kwargs...) = system

@inline function shift_particles!(system::AbstractFluidSystem, refinement, v_ode, u_ode,
                                  semi; iter=3)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # u_ode_initial = copy(u_ode)
    # v_ode_initial = copy(v_ode)
    u_initial = copy(u)

    # TrixiParticles.@autoinfiltrate    

    for i in 1:iter
        # Calculate the shift
        compute_shifting_haftu!(system, refinement, v, u, v_ode, u_ode, semi)
        # compute_shifting_adapted!(system, refinement, v, u, v_ode, u_ode, semi)

        # TrixiParticles.@autoinfiltrate 

        # Apply the shifting
        apply_shift!(u, u_ode, system, semi; theta=0.1, use_dynamic_theta=true,
                     theta_ratio=0.25)

        # TrixiParticles.@autoinfiltrate 
    end

    # TrixiParticles.@autoinfiltrate 

    compute_gradient_haftu!(system, refinement, v, u, v_ode, u_ode, semi)
    # compute_gradient_adapted!(system, refinement, v, u, v_ode, u_ode, semi)

    # Apply property corrections once using the total displacement
    apply_property_corrections!(v, u, u_initial, system, semi)

    return system
end

# Compared to the Haftu approaches, it evaluates the repulsive kernel using the local 
# particle spacing (dx_local) and strictly local smoothing length (h_a). It also scales 
# the displacement using the symmetric local density instead of a global reference density.
@fastpow function compute_shifting_adapted!(system, refinement::ParticleRefinement,
                                            v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; delta_v) = cache

    NDIMS = ndims(system)

    set_zero!(delta_v)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
        backend = semi.parallelization_backend

        @threaded semi for particle in eachparticle(system)
            particle_radius = compact_support(system, system, particle)

            @inbounds foreach_neighbor(system_coords, neighbor_coords,
                                       neighborhood_search, backend,
                                       particle;
                                       search_radius=particle_radius) do particle, neighbor,
                                                                         pos_diff, distance
                m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                rho_a = @inbounds current_density(v, system, particle)
                rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

                h_a = smoothing_length(system, particle)

                # Dynamically calculates local target spacing rather than using global dx
                dx_local = particle_spacing(system, refinement, particle)

                # Evaluates both kernels strictly with the local h_a (no inflection point xi used)
                kernel_weight = kernel(smoothing_kernel, distance, h_a)
                Wdx = kernel(smoothing_kernel, dx_local, h_a)
                grad_kernel_shift = kernel_grad(smoothing_kernel, pos_diff, distance, h_a)

                # Use symmetric local density (rho_a + rho_b)
                delta_r_ = -h_a^2 * (1 + (kernel_weight / Wdx)^4 * 2 / 10) * m_b /
                           (rho_a + rho_b) * grad_kernel_shift

                for i in eachindex(delta_r_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds delta_v[idx] += delta_r_[i]
                end
            end
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi)

    return system
end

# Implements Haftu's approach.
# Unlike the adapted version above, it uses the kernel inflection point (xi) 
# and global reference density. Unlike the original Haftu version below, it mathematically fixes 
# the kernel evaluation by using xi * h_ab as the target distance.
@fastpow function compute_shifting_haftu!(system, refinement::ParticleRefinement,
                                          v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; delta_v) = cache

    NDIMS = ndims(system)

    set_zero!(delta_v)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
        backend = semi.parallelization_backend

        @threaded semi for particle in eachparticle(system)
            particle_radius = compact_support(system, system, particle)

            @inbounds foreach_neighbor(system_coords, neighbor_coords,
                                       neighborhood_search, backend,
                                       particle;
                                       search_radius=particle_radius) do particle, neighbor,
                                                                         pos_diff, distance
                m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

                h_a = smoothing_length(system, particle)
                h_b = smoothing_length(neighbor_system, neighbor)
                h_ab = 0.5 * (h_a + h_b)

                xi = kernel_inflection_point(smoothing_kernel)

                # Evaluates the shifting kernel symmetrically using h_ab
                W_rij = kernel(smoothing_kernel, distance, h_ab)

                # Evaluates at target distance (xi * h_ab) using standard smoothing length (h_ab)
                W_dx_xi = kernel(smoothing_kernel, xi * h_ab, h_ab)

                # Symmetric kernel gradient 
                grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

                # Uses the global reference density explicitly dictated by Eq. 15
                rho0 = system.particle_refinement.reference_density

                repulsion_factor = 1.0 + 0.24 * (W_rij / W_dx_xi)^4

                # Eq. 15
                delta_r_ = -0.5 * h_a^2 * (m_b / rho0) * repulsion_factor * grad_kernel

                # Write into the buffers 
                for i in eachindex(delta_r_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds delta_v[idx] += delta_r_[i]
                end
            end
        end
    end

    modify_shifting_at_free_surfaces!(system, u, semi)

    return system
end

@fastpow function compute_gradient_adapted!(system, refinement::ParticleRefinement,
                                            v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; grad_density, grad_velocity, grad_pressure) = cache

    NDIMS = ndims(system)
    set_zero!(grad_density)
    set_zero!(grad_velocity)
    set_zero!(grad_pressure)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
        backend = semi.parallelization_backend

        @threaded semi for particle in eachparticle(system)
            particle_radius = compact_support(system, system, particle)

            @inbounds foreach_neighbor(system_coords, neighbor_coords,
                                       neighborhood_search, backend,
                                       particle;
                                       search_radius=particle_radius) do particle, neighbor,
                                                                         pos_diff, distance
                m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                rho_a = @inbounds current_density(v, system, particle)
                rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

                h_a = smoothing_length(system, particle)
                h_b = smoothing_length(neighbor_system, neighbor)
                h_ab = 0.5 * (h_a + h_b)

                p_a = @inbounds current_pressure(v, system, particle)
                p_b = @inbounds current_pressure(v_neighbor, neighbor_system, neighbor)

                # Evaluate symmetrically to conserve mass and momentum
                grad_kernel_sym = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

                # Compute density gradient 
                grad_density_ = (m_b / rho_b) * (rho_b - rho_a) * grad_kernel_sym
                for i in eachindex(grad_density_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds grad_density[idx] += grad_density_[i]
                end

                # Compute velocity gradient 
                v_a = current_velocity(v, system, particle)
                v_b = current_velocity(v_neighbor, neighbor_system, neighbor)
                grad_velocity_ = (m_b / rho_b) * (v_b - v_a) * (grad_kernel_sym)'

                for j in 1:NDIMS
                    for i in 1:NDIMS
                        idx_3d = linear_index_matrix(i, j, particle, NDIMS)
                        @inbounds grad_velocity[idx_3d] += grad_velocity_[i, j]
                    end
                end

                # Compute pressure gradient
                grad_pressure_ = (m_b / rho_b) * (p_b - p_a) * grad_kernel_sym
                for i in eachindex(grad_pressure_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds grad_pressure[idx] += grad_pressure_[i]
                end
            end
        end
    end

    return system
end

@fastpow function compute_gradient_haftu!(system, refinement::ParticleRefinement,
                                          v, u, v_ode, u_ode, semi)
    (; cache, smoothing_kernel) = system
    (; grad_density, grad_velocity, grad_pressure) = cache
    NDIMS = ndims(system)

    set_zero!(grad_density)
    set_zero!(grad_velocity)
    set_zero!(grad_pressure)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)
        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
        backend = semi.parallelization_backend
        @threaded semi for particle in eachparticle(system)
            particle_radius = compact_support(system, system, particle)

            @inbounds foreach_neighbor(system_coords, neighbor_coords,
                                       neighborhood_search, backend,
                                       particle;
                                       search_radius=particle_radius) do particle, neighbor,
                                                                         pos_diff, distance
                m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

                h_a = smoothing_length(system, particle)
                h_b = smoothing_length(neighbor_system, neighbor)
                h_ab = 0.5 * (h_a + h_b)

                # Symmetric kernel gradient 
                grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

                rho_a = @inbounds current_density(v, system, particle)
                rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

                p_a = @inbounds current_pressure(v, system, particle)
                p_b = @inbounds current_pressure(v_neighbor, neighbor_system, neighbor)

                # Compute density gradient 
                grad_density_ = (m_b / rho_b) * (rho_b - rho_a) * grad_kernel
                for i in eachindex(grad_density_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds grad_density[idx] += grad_density_[i]
                end

                # Compute velocity gradient
                v_a = current_velocity(v, system, particle)
                v_b = current_velocity(v_neighbor, neighbor_system, neighbor)
                grad_velocity_ = (m_b / rho_b) * (v_b - v_a) * (grad_kernel)'

                for j in 1:NDIMS
                    for i in 1:NDIMS
                        idx_3d = linear_index_matrix(i, j, particle, NDIMS)
                        @inbounds grad_velocity[idx_3d] += grad_velocity_[i, j]
                    end
                end

                # Compute pressure gradient
                grad_pressure_ = (m_b / rho_b) * (p_b - p_a) * grad_kernel
                for i in eachindex(grad_pressure_)
                    idx = linear_index_vector(i, particle, NDIMS)
                    @inbounds grad_pressure[idx] += grad_pressure_[i]
                end
            end
        end
    end

    return system
end

@inline function dynamic_theta(particle, u, delta_v, system, NDIMS; theta_ratio)
    shift_norm = zero(eltype(u))
    for i in 1:NDIMS
        idx = linear_index_vector(i, particle, NDIMS)
        shift_norm += delta_v[idx]^2
    end
    shift_norm = sqrt(shift_norm)

    h_i = smoothing_length(system, particle)

    # # Formulation exactly as in the papers source code
    # r_allowed = 0.25 * 0.025 * h_i

    r_allowed = theta_ratio * h_i

    return shift_norm > r_allowed ? (r_allowed / shift_norm) : 1.0
end

function apply_shift!(u_system, u_ode, system, semi; theta, use_dynamic_theta, theta_ratio)
    (; delta_v) = system.cache
    NDIMS = ndims(system)

    if use_dynamic_theta
        @threaded semi for particle in eachparticle(system)
            theta = dynamic_theta(particle, u_system, delta_v, system, NDIMS; theta_ratio)
            for i in 1:NDIMS
                idx = linear_index_vector(i, particle, NDIMS)
                @inbounds u_system[i, particle] += theta * delta_v[idx]
            end
        end
    else
        @threaded semi for particle in eachparticle(system)
            for i in 1:NDIMS
                idx = linear_index_vector(i, particle, NDIMS)
                @inbounds u_system[i, particle] += theta * delta_v[idx]
            end
        end
    end

    foreach_system(semi) do neighbor
        u_neighbor = wrap_u(u_ode, neighbor, semi)
        neighborhood_search = get_neighborhood_search(system, neighbor, semi)

        update_nhs!(neighborhood_search, system, neighbor, u_system, u_neighbor, semi)
    end
end

function apply_property_corrections!(v, u, u_initial, system, semi)
    (; cache) = system
    (; grad_density, grad_velocity, grad_pressure) = cache
    NDIMS = ndims(system)
    velocity = current_velocity(v, system)

    @threaded semi for particle in eachparticle(system)
        for i in 1:NDIMS
            # Extract the total shift dr_i across all iterations
            @inbounds dr_i = u[i, particle] - u_initial[i, particle]
            idx = linear_index_vector(i, particle, NDIMS)

            # Eq 37: Taylor series correction for Density
            @inbounds current_density(v, system)[particle] += grad_density[idx] * dr_i

            # Eq 37: Taylor series correction for Pressure
            @inbounds current_pressure(v, system)[particle] += grad_pressure[idx] * dr_i

            # Eq 37: Taylor series correction for Velocity
            for j in 1:NDIMS
                idx_3d = linear_index_matrix(j, i, particle, NDIMS)
                @inbounds velocity[j, particle] += grad_velocity[idx_3d] * dr_i
            end
        end
    end
end

# Fallback for kernels that do not have a inflection point implemented yet
function kernel_inflection_point(::AbstractSmoothingKernel)
    error("Inflection point not defined for this kernel.")
end

@inline kernel_inflection_point(::SchoenbergQuinticSplineKernel) = 0.759298480738450
@inline kernel_inflection_point(::WendlandC2Kernel) = 0.5
