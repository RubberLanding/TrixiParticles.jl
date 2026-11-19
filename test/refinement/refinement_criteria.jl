@testset verbose=true "Refinement Criteria" begin
    particle_spacing = 0.1
    n_particles = 10
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257.0

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    # The wall spacing is `particle_spacing / 2 = 0.05`
    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=2, faces=(true, true, true, false),
                           spacing_ratio=2)
    wall_spacing = tank.boundary.particle_spacing

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function setup(refinement_criteria; kwargs...)
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                       min_spacing=wall_spacing,
                                                       max_spacing=particle_spacing,
                                                       reference_density=density,
                                                       smoothing_length_factor=1.5,
                                                       refinement_criteria)

        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length, state_equation,
                                                   density_calculator=SummationDensity(),
                                                   particle_refinement=refinement,
                                                   kwargs...)

        semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search)
        ode = semidiscretize(semi, (0, 0))
        v_ode, u_ode = ode.u0.x

        return fluid_system, ode.p.semi, v_ode, u_ode
    end

    # Fluid particles closer to the wall than the kernel support
    y = tank.fluid.coordinates[2, :]
    x = tank.fluid.coordinates[1, :]
    distance_to_wall = min.(x, width .- x, y)
    near_wall = distance_to_wall .< 0.5 * particle_spacing
    far_from_wall = distance_to_wall .> 3 * smoothing_length

    solution_criterion(target) = TrixiParticles.SolutionRefinementCriterion((system,
                                                                             particle,
                                                                             v, u,
                                                                             semi) -> target)

    @testset "Default Criterion" begin
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                       min_spacing=wall_spacing,
                                                       max_spacing=particle_spacing,
                                                       reference_density=density)

        @test refinement.criteria == (TrixiParticles.SpatialRefinementCriterion(),)
    end

    @testset "Spatial Criterion" begin
        fluid_system, semi, v_ode,
        u_ode = setup(TrixiParticles.SpatialRefinementCriterion())
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(is_fixed[near_wall])
        @test all(target_spacing[near_wall] .== wall_spacing)

        @test !any(is_fixed[far_from_wall])
        @test all(target_spacing[far_from_wall] .== particle_spacing)

        # The spacing of the previous call is overwritten near boundaries
        # and kept everywhere else
        target_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(target_spacing[near_wall] .== wall_spacing)
        @test all(target_spacing[far_from_wall] .== 0.02)
    end

    @testset "Solution Criterion Finer Than Boundary" begin
        target = 0.03

        # The solution criterion is listed first, but the spatial criterion is applied first
        fluid_system, semi, v_ode,
        u_ode = setup((solution_criterion(target),
                       TrixiParticles.SpatialRefinementCriterion()))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # As in the code of Haftu et al., the solution criterion does not apply to
        # particles near boundaries, so they keep the boundary spacing
        @test all(target_spacing[near_wall] .== wall_spacing)

        # All other particles take the target spacing and are fixed
        @test all(is_fixed)
        @test all(target_spacing[far_from_wall] .== target)

        # A finer spacing of the previous call is kept away from boundaries,
        # but overwritten by the boundary spacing near boundaries
        target_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)
        @test all(target_spacing[far_from_wall] .== 0.02)
        @test all(target_spacing[near_wall] .== wall_spacing)
    end

    @testset "Solution Criterion Coarser Than Boundary" begin
        # Coarser than the wall spacing, but finer than `max_spacing`
        target = 0.08

        fluid_system, semi, v_ode,
        u_ode = setup((TrixiParticles.SpatialRefinementCriterion(),
                       solution_criterion(target)))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(target_spacing[near_wall] .== wall_spacing)
        @test all(target_spacing[far_from_wall] .== target)
        @test all(is_fixed)
    end

    @testset "Solution Criterion Without Spatial Criterion" begin
        target = 0.03
        fluid_system, semi, v_ode, u_ode = setup(solution_criterion(target))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # Without a spatial criterion, the boundary spacing is not used
        @test all(is_fixed)
        @test all(target_spacing .== target)
    end

    @testset "Multiple Solution Criteria" begin
        # The criterion for the particles with `x < 0.5` requests `max_spacing`,
        # so it is only active for the particles with `x >= 0.5`
        region_criterion = TrixiParticles.SolutionRefinementCriterion((system, particle,
                                                                       v, u,
                                                                       semi) -> u[1,
                                                                                  particle] <
                                                                                0.5 ?
                                                                                particle_spacing :
                                                                                0.03)

        fluid_system, semi, v_ode,
        u_ode = setup((solution_criterion(0.08), region_criterion))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # The smallest requested spacing is used
        @test all(is_fixed)
        @test all(target_spacing[x .< 0.5] .== 0.08)
        @test all(target_spacing[x .>= 0.5] .== 0.03)
    end

    @testset "Inactive Solution Criterion" begin
        fluid_system, semi, v_ode, u_ode = setup(solution_criterion(particle_spacing))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # A target spacing of `max_spacing` does not fix any particle
        @test !any(is_fixed)
        @test all(target_spacing .== particle_spacing)
    end

    @testset "Inactive Particles Are Skipped" begin
        fluid_system, semi, v_ode,
        u_ode = setup((TrixiParticles.SpatialRefinementCriterion(),
                       solution_criterion(0.03)); buffer_size=2)
        (; target_spacing, is_fixed) = fluid_system.cache
        n_fluid = nparticles(tank.fluid)
        inactive = (n_fluid + 1):(n_fluid + 2)

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(is_fixed[1:n_fluid])
        @test !any(is_fixed[inactive])
        @test all(target_spacing[inactive] .== particle_spacing)
    end

    @testset verbose=true "Vorticity Criterion" begin
        n_fluid = nparticles(tank.fluid)

        # Particles whose compact support is inside of the fluid
        inner = min.(distance_to_wall, height .- y) .> 2 * smoothing_length

        # Set the velocity of the fluid particles to `velocity_function(x)`.
        # The summation density is only computed in the right-hand side, so set it here.
        function set_velocity!(fluid_system, semi, v_ode, u_ode, velocity_function)
            v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
            u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
            fluid_system.cache.density .= density

            for particle in 1:n_fluid
                x_ = TrixiParticles.current_coords(u, fluid_system, particle)
                v[:, particle] .= velocity_function(x_)
            end

            return fluid_system
        end

        function setup_vorticity(velocity_function, refinement_criteria; kwargs...)
            fluid_system, semi, v_ode, u_ode = setup(refinement_criteria; kwargs...)
            set_velocity!(fluid_system, semi, v_ode, u_ode, velocity_function)

            return fluid_system, semi, v_ode, u_ode
        end

        shear_rate = 2.0
        shear_flow(x) = SVector(shear_rate * x[2], 0.0)

        angular_velocity = 3.0
        rotation(x) = angular_velocity * SVector(-(x[2] - 0.5), x[1] - 0.5)

        # The z-component of the curl is `-shear_rate` for the shear flow
        # and `2 * angular_velocity` for the rotation
        @testset "Vorticity $name" for (name, velocity_function,
                                        expected) in (("Shear Flow", shear_flow,
                                         -shear_rate),
                                        ("Rotation", rotation,
                                         2 * angular_velocity))
            fluid_system, semi,
            v_ode,
            u_ode = setup_vorticity(velocity_function,
                                    VorticityRefinementCriterion(); buffer_size=2)
            (; vorticity) = fluid_system.cache

            TrixiParticles.compute_vorticity!(fluid_system, v_ode, u_ode, semi)

            @test all(isapprox.(vorticity[1:n_fluid][inner], expected, rtol=0.01))

            # Inactive particles have no vorticity
            @test iszero(vorticity[(n_fluid + 1):end])
        end

        @testset "Walls Do Not Contribute" begin
            fluid_system, semi,
            v_ode, u_ode = setup_vorticity(rotation, VorticityRefinementCriterion())
            TrixiParticles.compute_vorticity!(fluid_system, v_ode, u_ode, semi)

            # The same fluid without the wall
            refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                           min_spacing=wall_spacing,
                                                           max_spacing=particle_spacing,
                                                           reference_density=density,
                                                           smoothing_length_factor=1.5,
                                                           refinement_criteria=VorticityRefinementCriterion())
            fluid_only = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                     smoothing_length, state_equation,
                                                     density_calculator=SummationDensity(),
                                                     particle_refinement=refinement)
            ode_fluid_only = semidiscretize(Semidiscretization(fluid_only;
                                                               neighborhood_search),
                                            (0, 0))
            semi_fluid_only = ode_fluid_only.p.semi
            fluid_only = semi_fluid_only.systems[1]
            v_ode_fluid_only, u_ode_fluid_only = ode_fluid_only.u0.x
            set_velocity!(fluid_only, semi_fluid_only, v_ode_fluid_only, u_ode_fluid_only,
                          rotation)

            TrixiParticles.compute_vorticity!(fluid_only, v_ode_fluid_only,
                                              u_ode_fluid_only, semi_fluid_only)

            @test fluid_system.cache.vorticity ≈ fluid_only.cache.vorticity
        end

        @testset "Vorticity 3D" begin
            spacing_3d = 0.1
            fluid = RectangularShape(spacing_3d, (8, 8, 8), (0.0, 0.0, 0.0); density)
            refinement_3d = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                              min_spacing=spacing_3d / 2,
                                                              max_spacing=spacing_3d,
                                                              reference_density=density,
                                                              smoothing_length_factor=1.5,
                                                              refinement_criteria=VorticityRefinementCriterion())
            fluid_system = WeaklyCompressibleSPHSystem(fluid;
                                                       smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                                       smoothing_length=1.5 * spacing_3d,
                                                       state_equation,
                                                       density_calculator=SummationDensity(),
                                                       particle_refinement=refinement_3d)
            ode = semidiscretize(Semidiscretization(fluid_system), (0, 0))
            semi = ode.p.semi
            fluid_system = semi.systems[1]
            v_ode, u_ode = ode.u0.x
            fluid_system.cache.density .= density

            # Rigid rotation with the angular velocity `angular_velocity_3d`,
            # whose vorticity is `2 * angular_velocity_3d`
            angular_velocity_3d = SVector(1.0, -2.0, 3.0)
            v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
            for particle in TrixiParticles.eachparticle(fluid_system)
                x_ = SVector{3}(fluid.coordinates[:, particle])
                v[:, particle] .= cross(angular_velocity_3d, x_ .- 0.4)
            end

            TrixiParticles.compute_vorticity!(fluid_system, v_ode, u_ode, semi)
            vorticity = reshape(fluid_system.cache.vorticity, 3, :)

            # Particles whose compact support is inside of the fluid
            inner_3d = vec(all(0.3 .< fluid.coordinates .< 0.5, dims=1))
            @test count(inner_3d) == 8

            # The discretization error is about 1 % in 3D
            for particle in findall(inner_3d)
                @test isapprox(vorticity[:, particle], 2 * angular_velocity_3d,
                               rtol=0.02)
            end
        end

        @testset "Smoothed Vorticity" begin
            fluid_system, semi,
            v_ode, u_ode = setup_vorticity(shear_flow, VorticityRefinementCriterion())
            (; vorticity, smoothed_vorticity) = fluid_system.cache

            # Only one particle has a (negative) vorticity
            vortex_particle = findfirst(inner)
            vorticity .= 0
            vorticity[vortex_particle] = -3.0

            TrixiParticles.smooth_vorticity!(fluid_system, u_ode, semi)

            # The particles in the compact support of the vortex particle,
            # including itself, have the vorticity magnitude of the vortex particle.
            # Skip the particles at the edge of the compact support.
            compact_support = 2 * smoothing_length
            distance = vec(sqrt.(sum(abs2,
                                     tank.fluid.coordinates .-
                                     tank.fluid.coordinates[:, vortex_particle],
                                     dims=1)))

            @test count(distance .< compact_support - 0.01) > 1
            @test all(smoothed_vorticity[distance .< compact_support - 0.01] .== 3)
            @test iszero(smoothed_vorticity[distance .> compact_support + 0.01])
        end

        @testset "Refines to Minimum Spacing" begin
            criterion = VorticityRefinementCriterion(outlet_cutoff=nothing)
            fluid_system, semi, v_ode, u_ode = setup_vorticity(shear_flow, criterion)
            (; target_spacing, is_fixed) = fluid_system.cache

            TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

            # The minimum spacing of the refinement is the wall spacing
            @test all(is_fixed)
            @test all(target_spacing .== wall_spacing)
        end

        @testset "Target Spacing and Spatial Criterion" begin
            criterion = VorticityRefinementCriterion(target_spacing=0.03,
                                                     outlet_cutoff=nothing)
            fluid_system, semi,
            v_ode,
            u_ode = setup_vorticity(shear_flow,
                                    (criterion,
                                     TrixiParticles.SpatialRefinementCriterion()))
            (; target_spacing, is_fixed) = fluid_system.cache

            TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

            # Particles near walls keep the wall spacing
            @test all(is_fixed)
            @test all(target_spacing[near_wall] .== wall_spacing)
            @test all(target_spacing[far_from_wall] .== 0.03)
        end

        @testset verbose=true "Outlet Cutoff" begin
            # The initial fluid particles are at `0.05, 0.15, ..., 0.95` in both directions,
            # so the cutoff is at 90 % of this extent along the flow direction
            cutoff = 0.05 + 0.9 * (0.95 - 0.05)

            # The flow direction does not need to be normalized
            @testset "$name" for (name, flow_direction,
                                  outlet_region) in (("Default", nothing, x .> cutoff),
                                  ("Positive x", (1.0, 0.0),
                                   x .> cutoff),
                                  ("Negative x", [-2.0, 0.0],
                                   x .< 1 - cutoff),
                                  ("Positive y", (0.0, 3.0),
                                   y .> cutoff))
                criterion = VorticityRefinementCriterion(; flow_direction)

                # The buffer particles must not affect the cutoff
                fluid_system, semi,
                v_ode, u_ode = setup_vorticity(shear_flow, criterion; buffer_size=2)
                (; target_spacing, is_fixed) = fluid_system.cache

                TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode,
                                                          semi)

                @test count(outlet_region) == 10
                @test all(is_fixed[1:n_fluid][.!outlet_region])
                @test !any(is_fixed[1:n_fluid][outlet_region])
                @test all(target_spacing[1:n_fluid][outlet_region] .== particle_spacing)
            end

            @test_throws "`flow_direction` must not be zero" VorticityRefinementCriterion(flow_direction=(0.0,
                                                                                                          0.0))
        end

        @testset "Absolute Threshold" begin
            # The vorticity of 1e-3 is below the default absolute threshold of 0.01
            weak_shear_flow(x) = SVector(1e-3 * x[2], 0.0)

            @testset "$name" for (name, velocity_function, absolute_threshold,
                                  refined) in (("Default", weak_shear_flow, 0.01, false),
                                  ("Lower Threshold", weak_shear_flow, 1e-4,
                                   true),
                                  ("Fluid at Rest", x -> SVector(0.0, 0.0),
                                   0.01, false))
                criterion = VorticityRefinementCriterion(; absolute_threshold,
                                                         outlet_cutoff=nothing)
                fluid_system, semi,
                v_ode, u_ode = setup_vorticity(velocity_function, criterion)

                TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode,
                                                          semi)

                @test all(==(refined), fluid_system.cache.is_fixed)
            end
        end

        @testset "Relative Threshold" begin
            # The vorticity `-2y` grows with the height
            parabolic_flow(x) = SVector(x[2]^2, 0.0)
            relative_threshold = 0.5
            criterion = VorticityRefinementCriterion(; relative_threshold,
                                                     outlet_cutoff=nothing)
            fluid_system, semi, v_ode, u_ode = setup_vorticity(parabolic_flow, criterion)
            (; smoothed_vorticity, is_fixed) = fluid_system.cache

            TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

            # Only particles above the relative threshold are refined
            threshold = relative_threshold * maximum(smoothed_vorticity)
            @test is_fixed == (smoothed_vorticity .> threshold)

            # The bottom of the fluid has a small vorticity, the top a large one
            @test !any(is_fixed[y .< 0.1])
            @test all(is_fixed[y .> 0.5])
        end
    end
end
