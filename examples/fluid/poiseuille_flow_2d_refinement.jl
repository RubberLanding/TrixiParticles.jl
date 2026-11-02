# ==========================================================================================
# 2D Poiseuille Flow Simulation with Particle Refinement (Entropically Damped SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 2D Poiseuille flow simulation in a rectangular channel
# including open boundary conditions.
# The fluid is refined near the walls and near the open boundaries.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
channel_height = 0.001 # distance between top and bottom walls
channel_length = 0.004 # distance between inlet and outlet

particle_spacing = (channel_height / 30) * 2

spacing_ratio = 2
min_spacing = particle_spacing / spacing_ratio
# This needs to equal the particle spacing, else we need
# to set the search radius in the nhs based on this max spacing!
max_spacing = particle_spacing

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled. The open boundaries have the minimum spacing of the refinement,
# so this is the same zone width as in the example without refinement.
open_boundary_layers = 10 * spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)

domain_size = (channel_length, channel_height)

open_boundary_size = (open_boundary_layers * min_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
imposed_pressure_drop = 0.1
outlet_pressure = 0.1
inlet_pressure = outlet_pressure + imposed_pressure_drop
const dynamic_viscosity = sqrt(fluid_density * channel_height^3 * imposed_pressure_drop /
                               (8 * channel_length * reynolds_number))

v_max = channel_height^2 * imposed_pressure_drop / (8 * dynamic_viscosity * channel_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0)

# The channel walls and the open boundaries have the minimum spacing of the refinement.
# They hold the fluid next to them at this spacing.
channel = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                          pressure=(pos) -> outlet_pressure +
                                            imposed_pressure_drop *
                                            (1 - (pos[1] / channel_length)),
                          n_layers=boundary_layers, faces=(false, false, true, true),
                          coordinates_eltype=Float64, spacing_ratio=spacing_ratio)

# The analytical solution uses the full inflow-to-outflow extent x in [0, channel_length].
# Since the pressure is prescribed at the free surface of each `BoundaryZone` instead of at
# its `boundary_face`, the inlet and outlet free surfaces lie at x = 0 and
# x = channel_length (= L), respectively. Thus, the boundary zones are effectively part of
# the computational domain in this setup.
inlet = RectangularTank(min_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers, pressure=inlet_pressure,
                        min_coordinates=(0.0, 0.0), faces=(false, false, true, true),
                        coordinates_eltype=Float64)

outlet = RectangularTank(min_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=(channel.fluid_size[1] - open_boundary_size[1],
                                          0.0),
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

fluid = setdiff(channel.fluid, inlet.fluid, outlet.fluid)

n_buffer_particles = 10 * channel.n_particles_per_dimension[2]^2

# ==========================================================================================
# ==== Fluid
smoothing_length_ratio = 2.0
smoothing_length = smoothing_length_ratio * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density

viscosity = ViscosityAdami(nu=kinematic_viscosity)

# `TransportVelocityAdami` is not adapted to multi-resolution yet
shifting_technique = ParticleShiftingTechnique()

state_equation = nothing

# ==========================================================================================
# ==== Refinement
inlet_face_x = open_boundary_size[1]
outlet_face_x = channel.fluid_size[1] - open_boundary_size[1]

# Refine the fluid next to the open boundaries, so that particles enter and leave
# the boundary zones with the open boundary spacing.
# The spatial criterion only considers walls, not open boundaries.
refinement_box_width = 8 * particle_spacing

open_boundary_logic = (system, particle, v, u,
                       semi) -> begin
    pos = TrixiParticles.current_coords(u, system, particle)
    if pos[1] < inlet_face_x + refinement_box_width ||
       pos[1] > outlet_face_x - refinement_box_width
        return min_spacing
    else
        return max_spacing
    end
end

spatial_criterion = SpatialRefinementCriterion()
open_boundary_criterion = SolutionRefinementCriterion(open_boundary_logic)
refinement_criteria = (spatial_criterion, open_boundary_criterion)

n_buffer_refinement_particles = 1000
n_buffer_particles += n_buffer_refinement_particles

# With `spacing_ratio=1.2`, it takes `log(2) / log(1.2) ≈ 4` neighborhoods
# to grade from `min_spacing` to `max_spacing`
refinement = ParticleRefinement(; spacing_ratio=1.2,
                                min_spacing, max_spacing,
                                smoothing_length_factor=smoothing_length_ratio,
                                reference_density=fluid_density,
                                refinement_criteria)

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity,
                                           density_calculator=fluid_density_calculator,
                                           shifting_technique,
                                           buffer_size=n_buffer_particles,
                                           particle_refinement=refinement)

# ==========================================================================================
# ==== Open Boundary
open_boundary_spacing = min_spacing
open_boundary_model = BoundaryModelDynamicalPressureZhang()

inlet_boundary_type = BidirectionalFlow()
inlet_face = ([inlet_face_x, 0.0], [inlet_face_x, channel.fluid_size[2]])
inlet_reference_velocity = nothing
inlet_reference_pressure = 0.2
inlet_boundary_zone = BoundaryZone(; boundary_face=inlet_face, face_normal=flow_direction,
                                   open_boundary_layers, density=fluid_density,
                                   particle_spacing=open_boundary_spacing,
                                   reference_velocity=inlet_reference_velocity,
                                   reference_pressure=inlet_reference_pressure,
                                   initial_condition=inlet.fluid,
                                   boundary_type=inlet_boundary_type)

outlet_boundary_type = BidirectionalFlow()
outlet_face = ([outlet_face_x, 0.0], [outlet_face_x, channel.fluid_size[2]])
outlet_reference_velocity = nothing
outlet_reference_pressure = 0.1
outlet_flow_direction = (.-(flow_direction))
outlet_boundary_zone = BoundaryZone(; boundary_face=outlet_face,
                                    face_normal=outlet_flow_direction,
                                    open_boundary_layers, density=fluid_density,
                                    particle_spacing=open_boundary_spacing,
                                    reference_velocity=outlet_reference_velocity,
                                    reference_pressure=outlet_reference_pressure,
                                    initial_condition=outlet.fluid,
                                    boundary_type=outlet_boundary_type)

open_boundary = OpenBoundarySystem(inlet_boundary_zone, outlet_boundary_zone; fluid_system,
                                   boundary_model=open_boundary_model,
                                   calculate_flow_rate=true,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall_boundary = union(channel.boundary)
min_smoothing_length = smoothing_length_ratio * min_spacing

boundary_model = BoundaryModelDummyParticles(wall_boundary.density, wall_boundary.mass,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             min_smoothing_length; state_equation,
                                             viscosity)

boundary_system = WallBoundarySystem(wall_boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall_boundary.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(wall_boundary.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(interval=1, prefix="", output_directory="out")

extra_callback = nothing

v_tmp = similar(ode.u0.x[1])
u_tmp = similar(ode.u0.x[2])
startup_done = Ref(false)
function apply_refinement!(integrator)
    v_ode = integrator.u.x[1]
    u_ode = integrator.u.x[2]
    semi = integrator.p.semi
    length_u_before = length(u_ode)

    # Build the graded spacing field before the first time step
    if !startup_done[]
        TrixiParticles.refinement_startup!(semi, v_ode, u_ode, v_tmp, u_tmp)
        startup_done[] = true
    end

    TrixiParticles.refinement!(semi, v_ode, u_ode, v_tmp, u_tmp)

    if length(u_ode) > length_u_before
        resize!(integrator,
                (length(v_ode), length(u_ode)))
    end

    derivative_discontinuity!(integrator, true)
end

last_refined_step = Ref(-1)
function refinement_condition(u, t, integrator)
    n = integrator.stats.naccept
    (n % 10 == 0 && n != last_refined_step[]) || return false
    last_refined_step[] = n
    return true
end

refinement_callback = DiscreteCallback(refinement_condition, apply_refinement!,
                                       save_positions=(false, false))

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback,
                        refinement_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
