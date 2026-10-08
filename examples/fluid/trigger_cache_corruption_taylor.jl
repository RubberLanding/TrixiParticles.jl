# ==========================================================================================
# 2D Taylor-Green Vortex Simulation (MWE for DtNaN on resize!)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqCore
using OrdinaryDiffEqLowStorageRK
using StaticArrays

# Added missing definitions to make the script executable
refine = true
refinement = nothing

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

function pressure_function(pos, t)
    x = pos[1]
    y = pos[2]
    return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
end

initial_pressure_function(pos) = pressure_function(pos, 0.0)

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

perturb_coordinates = true
fluid = RectangularShape(particle_spacing, (n_particles_xy, n_particles_xy), (0.0, 0.0),
                         coordinates_perturbation=perturb_coordinates ? 0.1 : nothing,
                         density=fluid_density, pressure=initial_pressure_function,
                         velocity=initial_velocity_function)

# Pre-allocate 1 buffer particle
n_buffer_particles = 1

viscosity = ViscosityAdami(; nu)
density_calculator = SummationDensity()

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator,
                                           viscosity,
                                           shifting_technique,
                                           particle_refinement=refinement,
                                           buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[box_length, box_length])

@inline function each_active_particle(system::AbstractFluidSystem, buffer)
        return 1:buffer.active_particle_count[]
    end

semi = Semidiscretization(fluid_system,
                          neighborhood_search=TrivialNeighborhoodSearch{2}(; periodic_box))

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(interval=1)
pp_callback = nothing

if refine
    refinement_callback = DiscreteCallback((u, t, integrator) -> integrator.stats.naccept == 1,
                                           (integrator) -> begin
                                                # EDAC systems integrate velocity (v), position (u), and pressure (p)
                                                v_ode = integrator.u.x[1]
                                                u_ode = integrator.u.x[2]
                                                p_ode = integrator.u.x[3]
                                               
                                                # Activate the 1 buffer particle by appending it to the state arrays.
                                                # Pushing it to 1e16 intentionally triggers the NaN.
                                                append!(v_ode, [0.0, 0.0])
                                                append!(u_ode, [1e16, 1e16])
                                                append!(p_ode, [0.0])

                                                # Dynamically extract lengths to handle the 3-part EDAC partition correctly
                                                new_sizes = length.(integrator.u.x)
                                                resize!(integrator, tuple(new_sizes...))
                                                
                                                # TrixiParticles.@autoinfiltrate

                                                # This forces f(du, u, p, t) to evaluate the new 1e16 particle immediately
                                                derivative_discontinuity!(integrator, true)
                                             end,
                                           save_positions=(false, false))
end

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), refinement_callback)

dt_max = min(smoothing_length / 4 * (sound_speed + U), smoothing_length^2 / (8 * nu))

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, 
            reltol=1e-4, 
            dtmax=dt_max, save_everystep=false, callback=callbacks);