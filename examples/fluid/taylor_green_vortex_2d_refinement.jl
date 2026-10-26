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
particle_spacing = 0.025
min_spacing = particle_spacing / 2
max_spacing = particle_spacing

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

smoothing_length_factor = 1.0
smoothing_length = smoothing_length_factor * particle_spacing
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

box_logic = (system, particle, v, u,
             semi) -> begin
    x_min = 0.35
    x_max = 0.65
    y_min = 0.35
    y_max = 0.65

    pos = TrixiParticles.current_coords(u, system, particle)
    if (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
        return min_spacing
    else 
        return max_spacing
    end 
end

box_criterion = SolutionRefinementCriterion(box_logic)
refinement_criteria = (box_criterion,)

refinement = ParticleRefinement(; spacing_ratio=1.05,
                                smoothing_length_factor,
                                min_spacing, max_spacing,
                                reference_density=fluid_density,
                                refinement_criteria)
n_childs = TrixiParticles.nchilds(refinement.splitting_pattern)

n_buffer_particles = 5000

viscosity = ViscosityAdami(; nu)
density_calculator = SummationDensity()
correction = nothing

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                            sound_speed,
                            density_calculator,
                            viscosity,
                            shifting_technique,
                            particle_refinement=refinement,
                            buffer_size=n_buffer_particles)
# ==========================================================================================
# ==== Simulation
min_corner=[0.0, 0.0]
max_corner=[box_length, box_length]
periodic_box = PeriodicBox(; min_corner, max_corner)
nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate(), periodic_box)

semi = Semidiscretization(fluid_system,
                          neighborhood_search=nhs)
# TrixiParticles.@autoinfiltrate

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(interval=1)

pp_callback = nothing

v_tmp = similar(ode.u0.x[1])
u_tmp = similar(ode.u0.x[2])
function apply_refinement!(integrator)
    v_ode = integrator.u.x[1]
    u_ode = integrator.u.x[2]
    semi = integrator.p.semi
    system = semi.systems[1]
    length_u_before = length(u_ode)

    integrator.stats.naccept > 3 && TrixiParticles.@autoinfiltrate    

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

    TrixiParticles.merge_particles!(system,
                                    v_ode, u_ode,
                                    semi)

    TrixiParticles.shift_particles!(system,
                                    v_ode, u_ode,
                                    semi)
    # TrixiParticles.@autoinfiltrate

    # Update the neighborhoodsearches of all systems
    TrixiParticles.update_nhs!(semi, u_ode)

    if length(u_ode) > length_u_before
        resize!(integrator,
                (length(v_ode), length(u_ode)))
        # TrixiParticles.@autoinfiltrate
    end

    derivative_discontinuity!(integrator, true)

    # integrator.stats.naccept > 5 && TrixiParticles.@autoinfiltrate    
end

refinement_callback = DiscreteCallback((u, t, integrator) -> integrator.stats.naccept % 10 == 0,
                                       apply_refinement!, save_positions=(false, false))

callbacks = CallbackSet(info_callback, saving_callback, pp_callback, UpdateCallback(),
                        refinement_callback)

dt_max = min(smoothing_length / 4 * (sound_speed + U), smoothing_length^2 / (8 * nu))

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=dt_max, save_everystep=false, callback=callbacks);

# using Plots
# function plot_points(system::AbstractFluidSystem, u) 
#     active_count = system.buffer.active_particle_count[]
#     masses = system.mass[1:active_count]
#     marker_sizes = sqrt.(masses ./ maximum(masses)) .* 8.0 

#     p1 = scatter(u[1, 1:active_count], u[2, 1:active_count], 
#                 markercolor=:blue, markerstrokewidth=0, markersize=marker_sizes, 
#                 legend=false, aspect_ratio=:equal)

#     plot(p1, size=(1500, 750))
# end

# function reset_ode!(u_ode_initial, v_ode_initial, u_ode, v_ode)
#     u_ode .= u_ode_initial
#     v_ode .= v_ode_initial
# end