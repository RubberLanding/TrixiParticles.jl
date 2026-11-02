# ==========================================================================================
# 2D Channel Flow Through a Periodic Array of Cylinders with Particle Refinement
#
# Based on:
#   S. Adami, X. Y. Hu, N. A. Adams.
#   "A transport-velocity formulation for smoothed particle hydrodynamics".
#   Journal of Computational Physics, Volume 241 (2013), pages 292-307.
#   https://doi.org/10.1016/J.JCP.2013.01.043
#
# This example simulates fluid flow driven by a body force through a channel
# containing a periodic array of cylinders. Due to periodicity, only one cylinder
# within a representative channel segment is simulated.
# The fluid is refined near the walls, near the cylinder and in the wake of the cylinder.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
n_particles_x = 144

spacing_ratio = 2

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# ==========================================================================================
# ==== Experiment Setup
acceleration_x = 2.5e-4

# Boundary geometry and initial fluid particle positions
cylinder_radius = 0.02
tank_size = (6 * cylinder_radius, 4 * cylinder_radius)
fluid_size = tank_size

reference_velocity = 1.2e-4
time_factor = cylinder_radius / reference_velocity
tspan = (0.0, 2.5 * time_factor)

fluid_density = 1000.0
nu = 0.1 / fluid_density # viscosity parameter

# Adami uses `c = 0.1 * sqrt(acceleration_x * cylinder_radius)`  but the original setup
# from M. Ellero and N. A. Adams (https://doi.org/10.1002/nme.3088) uses `c = 0.02`
sound_speed = 0.02

pressure = sound_speed^2 * fluid_density

particle_spacing = tank_size[1] / n_particles_x
min_spacing = particle_spacing / spacing_ratio
# This needs to equal the particle spacing, else we need
# to set the search radius in the nhs based on this max spacing!
max_spacing = particle_spacing

# The walls and the cylinder have the minimum spacing of the refinement.
# They hold the fluid next to them at this spacing.
box = RectangularTank(particle_spacing, fluid_size, tank_size, fluid_density;
                      n_layers=boundary_layers, pressure, faces=(false, false, true, true),
                      spacing_ratio)

cylinder_center = tank_size ./ 2
cylinder = SphereShape(min_spacing, cylinder_radius, cylinder_center,
                       fluid_density, sphere_type=RoundSphere())

fluid = setdiff(box.fluid, cylinder)
boundary = union(cylinder, box.boundary)

# ==========================================================================================
# ==== Fluid
smoothing_length_ratio = 2.0
smoothing_length = smoothing_length_ratio * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

state_equation = nothing

# ==========================================================================================
# ==== Refinement
# Refine the wake downstream of the cylinder.
# The box ends before the periodic boundary at `x = tank_size[1]`.
wake_logic = (system, particle, v, u,
              semi) -> begin
    x_min = cylinder_center[1]
    x_max = cylinder_center[1] + 2.5 * cylinder_radius
    y_min = cylinder_center[2] - cylinder_radius
    y_max = cylinder_center[2] + cylinder_radius

    pos = TrixiParticles.current_coords(u, system, particle)
    if (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
        return min_spacing
    else
        return max_spacing
    end
end

spatial_criterion = SpatialRefinementCriterion()
wake_criterion = SolutionRefinementCriterion(wake_logic)
refinement_criteria = (spatial_criterion, wake_criterion)

# The system is resized when more particles are needed
n_buffer_particles = 1000

# With `spacing_ratio=1.2`, it takes `log(2) / log(1.2) ≈ 4` neighborhoods
# to grade from `min_spacing` to `max_spacing`
refinement = ParticleRefinement(; spacing_ratio=1.2,
                                min_spacing, max_spacing,
                                smoothing_length_factor=smoothing_length_ratio,
                                reference_density=fluid_density,
                                refinement_criteria)

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator=ContinuityDensity(),
                                           viscosity=ViscosityAdami(; nu),
                                           shifting_technique=ParticleShiftingTechnique(),
                                           acceleration=(acceleration_x, 0.0),
                                           buffer_size=n_buffer_particles,
                                           particle_refinement=refinement)

# ==========================================================================================
# ==== Boundary
min_smoothing_length = smoothing_length_ratio * min_spacing

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             min_smoothing_length;
                                             viscosity=ViscosityAdami(; nu), state_equation)

boundary_system = WallBoundarySystem(boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
# The walls have `boundary_layers` layers with the minimum spacing
min_corner = (0.0, -boundary_layers * min_spacing)
max_corner = (tank_size[1], tank_size[2] + boundary_layers * min_spacing)
periodic_box = PeriodicBox(; min_corner, max_corner)
cell_list = FullGridCellList(; min_corner, max_corner)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box,
                                                                        cell_list))

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02 * time_factor, prefix="")

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

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        refinement_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            save_everystep=false, callback=callbacks);
