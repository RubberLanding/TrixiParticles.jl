@testset verbose=true "Cell Capacity" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
    particle_spacing = 0.1
    min_spacing = 0.025
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=2.0, min_spacing,
                                                   max_spacing=particle_spacing,
                                                   reference_density)

    tank = RectangularTank(particle_spacing, (0.5, 0.5), (1.0, 1.0), reference_density;
                           n_layers=3)

    function create_systems()
        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                     tank.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     1.2 * particle_spacing;
                                                     state_equation)

        fluid_system = EntropicallyDampedSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length=1.2 * particle_spacing,
                                                   sound_speed,
                                                   particle_refinement=refinement,
                                                   buffer_size=4)
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

        return fluid_system, boundary_system
    end

    function grid_nhs(; kwargs...)
        cell_list = FullGridCellList(; min_corner=(-0.5, -0.5), max_corner=(1.5, 1.5),
                                     kwargs...)

        # The default update strategy does not support inactive particles
        return GridNeighborhoodSearch{2}(; cell_list, update_strategy=SerialUpdate())
    end

    capacity(nhs) = PointNeighbors.max_points_per_cell(nhs.cell_list.cells)

    function capacities(semi)
        fluid_system, boundary_system = semi.systems

        nhs(system,
            neighbor) = TrixiParticles.get_neighborhood_search(system, neighbor,
                                                               semi)

        # The searches that store the refined fluid particles and the one that doesn't
        return (fluid_fluid=capacity(nhs(fluid_system, fluid_system)),
                boundary_fluid=capacity(nhs(boundary_system, fluid_system)),
                fluid_boundary=capacity(nhs(fluid_system, boundary_system)))
    end

    # Four times the number of particles of a uniform distribution at `min_spacing`
    function expected_capacity(semi)
        fluid_system = semi.systems[1]
        nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)

        return ceil(Int, 4 * prod(nhs.cell_size ./ min_spacing))
    end

    @testset verbose=true "$(nameof(handler))" for handler in (SharedNHSHandler,
                                                    PairsNHSHandler)
        @testset "Default Capacity" begin
            semi = Semidiscretization(create_systems()...;
                                      neighborhood_search=grid_nhs(),
                                      neighborhood_search_handler=handler,
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.1))
            v_ode, u_ode = ode.u0.x
            (; semi) = ode.p
            n_expected = expected_capacity(semi)

            # The default capacity is too small for the refined fluid
            @test n_expected > 100

            (; fluid_fluid, boundary_fluid, fluid_boundary) = capacities(semi)
            @test fluid_fluid == n_expected
            @test boundary_fluid == n_expected

            # The search that stores the boundary particles keeps the default capacity
            @test fluid_boundary == 100

            # Resizing recreates the neighborhood searches and must keep their capacity
            fluid_system, boundary_system = semi.systems
            TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                           [nparticles(fluid_system) + 7,
                                               nparticles(boundary_system)])
            @test capacities(semi) == (; fluid_fluid, boundary_fluid, fluid_boundary)
        end

        @testset "Larger Capacity Is Kept" begin
            semi = Semidiscretization(create_systems()...;
                                      neighborhood_search=grid_nhs(max_points_per_cell=1000),
                                      neighborhood_search_handler=handler,
                                      parallelization_backend=SerialBackend())

            @test expected_capacity(semi) < 1000
            @test capacities(semi) == (fluid_fluid=1000, boundary_fluid=1000,
                   fluid_boundary=1000)
        end
    end

    @testset "Without Refinement" begin
        fluid_system = EntropicallyDampedSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length=1.2 * particle_spacing,
                                                   sound_speed)
        semi = Semidiscretization(fluid_system; neighborhood_search=grid_nhs(),
                                  parallelization_backend=SerialBackend())
        nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)

        @test capacity(nhs) == 100
    end
end
