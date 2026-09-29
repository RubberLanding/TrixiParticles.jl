# ==========================================================================================
# 2D Vortex Street
#
# Based on:
#   A. Tafuni, J. M. Domínguez, R. Vacondio, A. J. C. Crespo.
#   "A versatile algorithm for the treatment of open boundary conditions in smoothed
#   particle hydrodynamics GPU models".
#   Computer Methods in Applied Mechanics and Engineering, Volume 342 (2018), pages 604-624.
#   https://doi.org/10.1016/j.cma.2018.08.004
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqCore

# ==========================================================================================
# ==== Resolution
particle_spacing_factor = 0.5 # Resolution in the paper is `0.01` (5M particles)

cylinder_diameter = 0.1
particle_spacing = particle_spacing_factor * cylinder_diameter

spacing_ratio = 2
min_spacing = particle_spacing / spacing_ratio

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`.
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 4.0)

# We use a smaller domain for a faster runtime. The original domain size in the paper is
# domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)
domain_size = (20 * cylinder_diameter, 10 * cylinder_diameter)
open_boundary_size = (particle_spacing * open_boundary_layers, domain_size[2])

flow_direction = SVector(1.0, 0.0)
reynolds_number = 200
prescribed_velocity = 1.0
fluid_density = 1000.0
# Maximum velocity observed in the vortex street is typically around 1.5
v_max = 1.5
sound_speed = 10 * v_max

pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density;
                       n_layers=boundary_layers,
                       velocity=prescribed_velocity * flow_direction,
                       faces=(false, false, true, true),
                       coordinates_eltype=Float64, spacing_ratio)

min_coords_inlet = (-open_boundary_layers * particle_spacing, 0.0)
inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density; n_layers=boundary_layers,
                        velocity=prescribed_velocity * flow_direction,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true),
                        coordinates_eltype=Float64, spacing_ratio)

min_coords_outlet = (pipe.fluid_size[1], 0.0)
outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         velocity=prescribed_velocity * flow_direction,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64, spacing_ratio)

n_buffer_inlet = nparticles(inlet.fluid)

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(min_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density; sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)


# ==========================================================================================
# ==== Fluid
smoothing_length_ratio = 1.5
smoothing_length = smoothing_length_ratio * particle_spacing
smoothing_length_min = smoothing_length_ratio * min_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = SummationDensity()

kinematic_viscosity = prescribed_velocity * cylinder_diameter / reynolds_number
viscosity = ViscosityAdami(nu=kinematic_viscosity)

background_pressure = sound_speed^2 * fluid_density
shifting_technique = ConsistentShiftingSun2019(; sound_speed_factor=0.1)

x_min = 0.9
x_max = 1.1
y_min = 0.4
y_max = 0.6

box_logic = (system, particle, v, u,
            semi) -> begin
    pos = TrixiParticles.current_coords(u, system, particle)
    return (x_min < pos[1] < x_max) && (y_min < pos[2] < y_max)
end

box_criterion = SolutionRefinementCriterion(box_logic)
spatial_criterion = SpatialRefinementCriterion()
refinement_criteria = (box_criterion, spatial_criterion)

refinement = ParticleRefinement(; n_particles=nparticles(fluid),
                                spacing_ratio=1.05,
                                min_spacing=particle_spacing / 2,
                                reference_density=fluid_density,
                                refinement_criteria)
n_childs = TrixiParticles.nchilds(refinement.splitting_pattern)

# Change 1.2 to 5 if you want to avoid a global resize in the first refinement
n_buffer_refinement = ceil(Int32,
                        1.2 * n_childs * (x_max - x_min) *
                        (y_max - y_min) / particle_spacing^2)
n_buffer_particles = n_buffer_inlet + n_buffer_refinement

refine = true

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator=fluid_density_calculator,
                                           viscosity,
                                           pressure_acceleration=nothing,
                                           shifting_technique,
                                           buffer_size=n_buffer_particles)

if refine
    fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                               sound_speed,
                                               density_calculator=fluid_density_calculator,
                                               viscosity,
                                               shifting_technique,
                                               particle_refinement=refinement,
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()

# Note that we use bidirectional flow for both the inlet and outlet.
# True inflow and outflow zones are only necessary for the Lastiwka model.
face_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_velocity=prescribed_velocity * flow_direction,
                      initial_condition=inlet.fluid)

face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], domain_size[2]])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       initial_condition=outlet.fluid)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary (EDAC Adaptation)
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)

# Remove `state_equation` keyword argument
boundary_model_wall = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length_min)

boundary_system_wall = WallBoundarySystem(wall, boundary_model_wall)

# Remove `state_equation` keyword argument, keep viscosity
boundary_model_cylinder = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                      AdamiPressureExtrapolation(),
                                                      smoothing_kernel, smoothing_length_min;
                                                      viscosity)

boundary_system_cylinder = WallBoundarySystem(cylinder, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(wall.coordinates .- particle_spacing, dims=2)
max_corner = maximum(wall.coordinates .+ particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system_wall,
                          boundary_system_cylinder; neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)

function system_smoothing_length(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return zero(eltype(system))
end 

function system_smoothing_length(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return view(system.smoothing_length, TrixiParticles.each_active_particle(system))
end 

saving_callback = SolutionSavingCallback(interval=1)
if refine
    saving_callback = SolutionSavingCallback(interval=1, my_custom_quantity=system_smoothing_length)
end 

extra_callback = nothing

refinement_callback = nothing

refine_interval = 0.05
next_refine_time = [0.0]

v_tmp = similar(ode.u0.x[1])
u_tmp = similar(ode.u0.x[2])

# TrixiParticles.@autoinfiltrate

function resize_integrator_cache!(integrator, semi, old_ranges_v, old_ranges_u)
    # Collect all history and intermediate caches
    caches_to_clean = [integrator.uprev, OrdinaryDiffEqCore.full_cache(integrator)..., 
                       integrator.fsalfirst, integrator.fsallast]
    
    # 1. SHIFT UPREV AND WIPE INTERMEDIATE CACHES
    for c in caches_to_clean
        # Skip invalid caches or caches of a different type
        if isnothing(c) || typeof(c) != typeof(integrator.u)
            continue
        end

        is_u = (c.x[1] === integrator.u.x[1])
        is_uprev = (c.x[1] === integrator.uprev.x[1])

        if is_uprev
            # `uprev` was resized but not shifted. We shift it right-to-left 
            # to safely push data into newly allocated space without overwriting.
            for sys_idx in reverse(eachindex(semi.systems))
                r_v_old = old_ranges_v[sys_idx]
                r_v_new = semi.ranges_v[sys_idx]
                
                r_u_old = old_ranges_u[sys_idx]
                r_u_new = semi.ranges_u[sys_idx]
                
                len_v = min(length(r_v_old), length(r_v_new))
                len_u = min(length(r_u_old), length(r_u_new))
                
                # Base.copyto! safely handles overlapping memory
                if len_v > 0
                    copyto!(c.x[1], first(r_v_new), c.x[1], first(r_v_old), len_v)
                end
                if len_u > 0
                    copyto!(c.x[2], first(r_u_new), c.x[2], first(r_u_old), len_u)
                end
            end
        elseif !is_u
            # INTERMEDIATE CACHES (k): Wipe completely to prevent boundary teleportation
            fill!(c.x[1], 0.0)
            fill!(c.x[2], 0.0)
        end
    end

    # 2. SANITIZE THE INACTIVE GAPS IN STATE ARRAYS
    for (sys_idx, sys) in enumerate(semi.systems)
        r_v_new = semi.ranges_v[sys_idx]
        r_u_new = semi.ranges_u[sys_idx]
        
        n_vars_v = size(TrixiParticles.wrap_v(integrator.u.x[1], sys, semi), 1)
        n_vars_u = size(TrixiParticles.wrap_u(integrator.u.x[2], sys, semi), 1)
        
        active_v_len = length(r_v_new)
        active_u_len = length(r_u_new)
        
        # Calculate true active lengths using the buffer[cite: 7]
        if hasproperty(sys, :buffer) && hasproperty(sys.buffer, :active_particle_count)
            active_v_len = sys.buffer.active_particle_count[] * n_vars_v
            active_u_len = sys.buffer.active_particle_count[] * n_vars_u
        end
        
        inactive_v_range = (first(r_v_new) + active_v_len):last(r_v_new)
        inactive_u_range = (first(r_u_new) + active_u_len):last(r_u_new)
        
        # Zero out the inactive gaps in both `u` and `uprev`
        for c in caches_to_clean
            if !isnothing(c) && typeof(c) == typeof(integrator.u)
                if c.x[1] === integrator.u.x[1] || c.x[1] === integrator.uprev.x[1]
                    if length(inactive_v_range) > 0
                        c.x[1][inactive_v_range] .= 0.0
                    end
                    if length(inactive_u_range) > 0
                        c.x[2][inactive_u_range] .= 0.0
                    end
                end
            end
        end
    end
end

function check_state_health(v_ode, system, step_name)
    p_max = maximum(abs, TrixiParticles.current_pressure(v_ode, system))
    v_max = maximum(abs, TrixiParticles.current_velocity(v_ode, system))
    println("After $step_name -> Max Pressure: $p_max | Max Velocity: $v_max")
end

if refine
    refinement_callback = DiscreteCallback((u, t, integrator) -> integrator.stats.naccept == 10,
                                           (integrator) -> begin
                                               v_ode = integrator.u.x[1]
                                               u_ode = integrator.u.x[2]
                                               semi = integrator.p.semi
                                               system = semi.systems[1]

                                             #   TrixiParticles.@autoinfiltrate

                                               length_u_before = length(u_ode)
                                               length_v_before = length(v_ode)
                                               old_ranges_v = copy(semi.ranges_v)
                                               old_ranges_u = copy(semi.ranges_u)

                                                # TrixiParticles.@autoinfiltrate

                                               TrixiParticles.reset_refinement!(system, semi)

                                               # Move active particles to the start and inactive particles to the end
                                               # Update self-interaction nhs
                                               TrixiParticles.compact_system!(system, v_ode, u_ode, semi)
                                               neighborhood_search = TrixiParticles.get_neighborhood_search(system,
                                                                                             semi)
                                               TrixiParticles.update_nhs!(neighborhood_search, system,
                                                           system, u_ode, u_ode, semi)

                                               # Apply refinement criterion 
                                               TrixiParticles.apply_refinement_criteria!(system, v_ode,
                                                                          u_ode, semi)

                                             #   TrixiParticles.@autoinfiltrate

                                               # Update the spacing of particles
                                               TrixiParticles.update_particle_spacing(system, v_ode, u_ode,
                                                                       semi)

                                             #   TrixiParticles.@autoinfiltrate

                                               # Split the particles
                                               TrixiParticles.split_particles!(semi, v_ode, u_ode, v_tmp,
                                                                u_tmp)

                                            #    TrixiParticles.@autoinfiltrate

                                               # Merge the particles
                                               TrixiParticles.merge_particles!(system, v_ode, u_ode, semi)

                                            #    TrixiParticles.@autoinfiltrate

                                               # Move active particles to the start and inactive particles to the end
                                               TrixiParticles.compact_system!(system, v_ode, u_ode, semi)
                                               
                                            #    TrixiParticles.@autoinfiltrate

                                               # After adding and deleting particles,
                                               # we need to update the searches for all systems
                                               TrixiParticles.update_nhs!(semi, u_ode)

                                            #    TrixiParticles.@autoinfiltrate

                                               # Update smoothing lengths
                                               TrixiParticles.update_smoothing_lengths!(system, v_ode,
                                                                         u_ode, semi)
                                                                        
                                            #    TrixiParticles.@autoinfiltrate

                                            #    # Shift the particles and correct the particle properties
                                               TrixiParticles.shift_particles!(system, v_ode, u_ode, semi,
                                                                integrator.dt)

                                            #    TrixiParticles.@autoinfiltrate

                                                # Expand the solver's internal history caches to match the new physical capacity
                                                if length(u_ode) > length_u_before
                                                    resize!(integrator, (length(v_ode), length(u_ode)))
                                                    resize_integrator_cache!(integrator, semi, old_ranges_v, old_ranges_u)
                                                end

                                                # Force the solver to natively recalculate the true fluid derivatives
                                                derivative_discontinuity!(integrator, true)
                                                
                                            #    TrixiParticles.@autoinfiltrate

                                               # next_refine_time[1] += refine_interval
                                             end,
                                           save_positions=(false, false))
end

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback,
                        refinement_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # May need tuning to prevent boundary penetration
            reltol=1e-4, # May need tuning to prevent boundary penetration
            save_everystep=false, callback=callbacks);
