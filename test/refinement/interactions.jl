@testset verbose=true "Variable Smoothing Length Interactions" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=7)
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5, min_spacing=0.05,
                                                   max_spacing=0.1,
                                                   reference_density=1000.0)

    # Evaluate the right-hand side for the given systems
    function rhs(systems...)
        semi = Semidiscretization(systems...; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        return parent(dv_ode), ode.p.semi
    end

    @testset verbose=true "Uniform Smoothing Length Matches Constant Smoothing Length" begin
        particle_spacing = 0.1
        smoothing_length = 1.2 * particle_spacing
        tank = RectangularTank(particle_spacing, (0.5, 0.5), (1.0, 1.0), 1000.0;
                               n_layers=3)

        # Use a non-trivial state, so that all terms contribute
        n_fluid = nparticles(tank.fluid)
        fluid = deepcopy(tank.fluid)
        fluid.velocity .= [sin(i + j) for i in 1:2, j in 1:n_fluid]
        fluid.density .+= [10 * cos(i) for i in 1:n_fluid]

        boundary_density_calculators = (ContinuityDensity(), SummationDensity(),
                                        AdamiPressureExtrapolation(),
                                        AdamiPressureExtrapolation(allow_loop_flipping=false))

        @testset "$(nameof(typeof(boundary_density_calculator)))" for boundary_density_calculator in
                                                                      boundary_density_calculators

            function dv(particle_refinement)
                viscosity = ViscosityAdami(nu=1e-3)
                fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                           smoothing_length, viscosity,
                                                           density_calculator=ContinuityDensity(),
                                                           state_equation,
                                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                                           particle_refinement)

                boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                             tank.boundary.mass,
                                                             boundary_density_calculator,
                                                             smoothing_kernel,
                                                             smoothing_length;
                                                             state_equation,
                                                             viscosity)
                boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

                return first(rhs(fluid_system, boundary_system))
            end

            dv_constant = dv(nothing)
            dv_uniform = dv(refinement)

            @test any(!iszero, dv_constant)
            @test isapprox(dv_uniform, dv_constant, rtol=1e-12)
        end
    end

    @testset verbose=true "Fluid-Fluid Pair" begin
        # The neighborhood search radius is 2 * 1.2 * 0.1 = 0.24
        coordinates = [0.0 0.15
                       0.0 0.0]
        mass = [1.0, 0.5]
        density = [1100.0, 1100.0]
        initial_condition = InitialCondition(; coordinates, mass, density,
                                             particle_spacing=0.1)

        function pair_dv(h_a, h_b)
            fluid_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            semi = Semidiscretization(fluid_system;
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.1))
            system = ode.p.semi.systems[1]
            TrixiParticles.set_particle_smoothing_length!(system, 1, h_a)
            TrixiParticles.set_particle_smoothing_length!(system, 2, h_b)

            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            return TrixiParticles.wrap_v(dv_ode, system, ode.p.semi)
        end

        @testset "Different Smoothing Lengths are Antisymmetric" begin
            dv = pair_dv(0.1, 0.12)

            @test !iszero(dv[1, 1])
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end

        @testset "Inside the Larger Compact Support Only" begin
            # The distance of 0.15 is outside of the compact support of the first particle
            dv = pair_dv(0.06, 0.12)

            @test !iszero(dv[1, 1])
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end

        @testset "Outside of Both Compact Supports" begin
            # The neighborhood search returns this pair, but it must not interact
            @test iszero(pair_dv(0.06, 0.06))
        end
    end

    @testset verbose=true "Wall Next to Fluid With Larger Smoothing Length" begin
        # The distance is outside of the compact support of the wall (2 * 0.06 = 0.12),
        # but inside of the shared compact support (2 * (0.06 + 0.12) / 2 = 0.18).
        fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[10.0], density=[1100.0],
                                 velocity=[1.0; 0.0;;], particle_spacing=0.1)
        wall = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[10.0], density=[1000.0],
                                particle_spacing=0.05)

        function create_systems(boundary_density_calculator)
            fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                         boundary_density_calculator,
                                                         smoothing_kernel, 0.06;
                                                         state_equation)

            return fluid_system, WallBoundarySystem(wall, boundary_model)
        end

        @testset "Search Radius" begin
            fluid_system, boundary_system = create_systems(ContinuityDensity())

            @test TrixiParticles.compact_support(boundary_system, fluid_system) ≈ 0.24

            # Without variable smoothing lengths, the search radius is unchanged
            constant_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                          smoothing_length=0.12,
                                                          density_calculator=ContinuityDensity(),
                                                          state_equation)
            @test TrixiParticles.compact_support(boundary_system, constant_system) ≈ 0.12
        end

        @testset "Continuity Equation" begin
            dv_ode, semi = rhs(create_systems(ContinuityDensity())...)
            dv_boundary = TrixiParticles.wrap_v(dv_ode, semi.systems[2], semi)

            # With an inactive wall gradient, only half of the fluid gradient remains
            grad_kernel = TrixiParticles.kernel_grad(smoothing_kernel, SVector(-0.15, 0.0),
                                                     0.15, 0.12) / 2
            expected = 10.0 * (1000.0 / 1100.0) * dot(SVector(-1.0, 0.0), grad_kernel)

            @test !iszero(dv_boundary[end, 1])
            @test isapprox(dv_boundary[end, 1], expected, rtol=1e-12)
        end

        @testset "Pressure Extrapolation Does Not Depend on the Loop Order" begin
            # With a single fluid and a single boundary particle and one thread,
            # `allow_loop_flipping=true` selects the loop over the fluid particles.
            pressure(allow_loop_flipping) = begin
                calculator = AdamiPressureExtrapolation(; allow_loop_flipping)
                _, semi = rhs(create_systems(calculator)...)
                only(semi.systems[2].boundary_model.pressure)
            end

            @test pressure(false) > 0
            @test isapprox(pressure(false), pressure(true), rtol=1e-12)
        end
    end
end
