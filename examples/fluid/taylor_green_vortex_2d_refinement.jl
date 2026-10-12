# ==========================================================================================
# 2D Taylor-Green Vortex Simulation
#
# Based on:
#   P. Ramachandran, K. Puri.
#   "Entropically damped artiﬁcial compressibility for SPH".
#   Computers and Fluids, Volume 179 (2019), pages 579-594.
#   https://doi.org/10.1016/j.compfluid.2018.11.023
#
# This example simulates the Taylor-Green vortex, a classic benchmark case for
# incompressible viscous flow, characterized by an array of decaying vortices.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqCore
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)
reynolds_number = 100.0

box_length = 1.0

U = 1.0 # m/s
fluid_density = 1.0
sound_speed = 10U

b = -8pi^2 / reynolds_number

# Taylor Green Vortex Pressure Function
function pressure_function(pos, t)
    x = pos[1]
    y = pos[2]

    return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
end

initial_pressure_function(pos) = pressure_function(pos, 0.0)

# Taylor Green Vortex Velocity Function
function velocity_function(pos, t)
    x = pos[1]
    y = pos[2]

    vel = U * exp(b * t) * [-cos(2pi * x) * sin(2pi * y), sin(2pi * x) * cos(2pi * y)]

    return SVector{2}(vel)
end

initial_velocity_function(pos) = velocity_function(pos, 0.0)

n_particles_xy = round(Int, box_length / particle_spacing)

# ==========================================================================================
# ==== Fluid
nu = U * box_length / reynolds_number

background_pressure = sound_speed^2 * fluid_density
shifting_technique = ConsistentShiftingSun2019(; sound_speed_factor=0.1)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

# To be set via `trixi_include`
perturb_coordinates = true
fluid = RectangularShape(particle_spacing, (n_particles_xy, n_particles_xy), (0.0, 0.0),
                         # Perturb particle coordinates to avoid stagnant streamlines without TVF
                         coordinates_perturbation=perturb_coordinates ? 0.1 : nothing, # To avoid stagnant streamlines when not using TVF.
                         density=fluid_density, pressure=initial_pressure_function,
                         velocity=initial_velocity_function)

refine = true
n_particles = TrixiParticles.nparticles(fluid)

x_min = 0.35
x_max = 0.65
y_min = 0.35
y_max = 0.65
box_logic = (system, particle, v, u,
             semi) -> begin
    pos = TrixiParticles.current_coords(u, system, particle)
    return (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
end

box_criterion = SolutionRefinementCriterion(box_logic)
refinement_criteria = (box_criterion,)

refinement = ParticleRefinement(; spacing_ratio=1.05,
                                min_spacing=particle_spacing / 1.5,
                                reference_density=fluid_density,
                                refinement_criteria)
n_childs = TrixiParticles.nchilds(refinement.splitting_pattern)

# n_buffer_particles = ceil(Int32,
#                         12 * n_childs * (x_max - x_min) *
#                         (y_max - y_min) / particle_spacing^2)
n_buffer_particles = 1000

viscosity = ViscosityAdami(; nu)
density_calculator = SummationDensity()
correction = nothing

fluid_system = refine ?
               EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator,
                                           viscosity,
                                           shifting_technique,
                                           particle_refinement=refinement,
                                           buffer_size=n_buffer_particles) :
               EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator,
                                           viscosity,
                                           pressure_acceleration=nothing,
                                           shifting_technique,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[box_length, box_length])
semi = Semidiscretization(fluid_system,
                          neighborhood_search=TrivialNeighborhoodSearch{2}(; periodic_box))
# TrixiParticles.@autoinfiltrate

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(interval=1)

pp_callback = nothing

refinement_callback = nothing
v_tmp = similar(ode.u0.x[1])
u_tmp = similar(ode.u0.x[2])

function apply_refinement!(integrator)
    v_ode = integrator.u.x[1]
    u_ode = integrator.u.x[2]
    semi = integrator.p.semi
    system = semi.systems[1]
    length_u_before = length(u_ode)

    TrixiParticles.reset_refinement!(system,
                                     semi)
    TrixiParticles.compact_system!(system, v_ode,
                                   u_ode, semi)
    neighborhood_search = TrixiParticles.get_neighborhood_search(system,
                                                                 semi)
    u = TrixiParticles.wrap_u(u_ode, system,
                              semi)
    TrixiParticles.update_nhs!(neighborhood_search,
                               system,
                               system, u, u,
                               semi)

    TrixiParticles.apply_refinement_criteria!(system,
                                              v_ode,
                                              u_ode,
                                              semi)

    TrixiParticles.update_particle_spacing(system,
                                           v_ode,
                                           u_ode,
                                           semi)

    TrixiParticles.split_particles!(semi, v_ode,
                                    u_ode, v_tmp,
                                    u_tmp)
    TrixiParticles.compact_system!(system, v_ode,
                                   u_ode, semi)
    TrixiParticles.update_nhs!(semi, u_ode)

    TrixiParticles.merge_particles!(system,
                                    v_ode, u_ode,
                                    semi)
    TrixiParticles.compact_system!(system, v_ode,
                                   u_ode, semi)
    TrixiParticles.update_nhs!(semi, u_ode)

    TrixiParticles.shift_particles!(system,
                                    v_ode, u_ode,
                                    semi,
                                    integrator.dt, 1.0, 5)

    TrixiParticles.update_smoothing_lengths!(system,
                                             v_ode,
                                             u_ode,
                                             semi)

    if length(u_ode) > length_u_before
        resize!(integrator,
                (length(v_ode), length(u_ode)))
        TrixiParticles.@autoinfiltrate
    end

    derivative_discontinuity!(integrator, true)
end

refinement_callback = refine ?
                      DiscreteCallback((u, t,
                                        integrator) -> integrator.stats.naccept == 5,
                                       apply_refinement!, save_positions=(false, false)) :
                      nothing

save_interval = 0.05
next_save_time = [0.0]
frame_counter = [0] # Integer frame counter
vtk_callback = DiscreteCallback((u, t, integrator) -> t >= next_save_time[1],
                                (integrator) -> begin
                                    (; t) = integrator
                                    v_ode = integrator.u.x[1]
                                    u_ode = integrator.u.x[2]
                                    semi = integrator.p.semi
                                    system = semi.systems[1]

                                    interpolation_min = [0.0, 0.0]
                                    interpolation_max = [box_length, box_length]
                                    interpolation_spacing = min_spacing / 4

                                    # Put the text before the number for automatic Paraview grouping and pad the counter to 4 digits
                                    refine_str = refine ? "ref" : "noref"
                                    frame_str = lpad(frame_counter[1], 4, '0')
                                    file_name = "taylor_greene_$(refine_str)_$(frame_str)"

                                    TrixiParticles.interpolate_plane_2d_vtk(interpolation_min,
                                                                            interpolation_max,
                                                                            interpolation_spacing,
                                                                            semi, system,
                                                                            v_ode, u_ode;
                                                                            filename=file_name)

                                    # Increment both the timer and the frame counter
                                    next_save_time[1] += save_interval
                                    frame_counter[1] += 1
                                end,
                                save_positions=(false, false))

callbacks = CallbackSet(info_callback, saving_callback, pp_callback, UpdateCallback(),
                        refinement_callback)

dt_max = min(smoothing_length / 4 * (sound_speed + U), smoothing_length^2 / (8 * nu))

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=dt_max, save_everystep=false, callback=callbacks);
