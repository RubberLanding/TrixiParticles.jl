# ==========================================================================================
# 2D Vortex Street with Particle Refinement
#
# Flow past a cylinder in a channel with open boundaries, based on `vortex_street_2d.jl`.
# The walls and the open boundaries have the coarse resolution, only the cylinder has
# the fine resolution. The fluid is refined near the cylinder (spatial criterion) and in
# the wake behind the cylinder, where the vorticity is large (vorticity criterion,
# solution adaptivity in Haftu et al.). Between the refined regions and the coarse fluid,
# the spacing is graded by the refinement.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
cylinder_diameter = 0.1

# Coarse resolution of the free stream
particle_spacing_factor = 0.2
particle_spacing = particle_spacing_factor * cylinder_diameter

spacing_ratio = 2
min_spacing = particle_spacing / spacing_ratio
# This needs to equal the particle spacing, else we need
# to set the search radius in the nhs based on this max spacing!
max_spacing = particle_spacing

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 4.0)

# We use a smaller domain for a faster runtime. The original domain size in the paper is
# domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)
domain_size = (20 * cylinder_diameter, 10 * cylinder_diameter)

flow_direction = SVector(1.0, 0.0)
reynolds_number = 200
const prescribed_velocity = 1.0
fluid_density = 1000.0
# Maximum velocity observed in the vortex street is typically around 1.5
v_max = 1.5
sound_speed = 10 * v_max

# The walls and the open boundaries have the coarse spacing and the cylinder has the
# minimum spacing of the refinement. The spatial criterion holds the fluid next to them
# at their spacing.
pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density;
                       n_layers=boundary_layers,
                       velocity=prescribed_velocity * flow_direction,
                       faces=(false, false, true, true),
                       coordinates_eltype=Float64)

open_boundary_size = (particle_spacing * open_boundary_layers, domain_size[2])

min_coords_inlet = (-open_boundary_layers * particle_spacing, 0.0)
inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density; n_layers=boundary_layers,
                        velocity=prescribed_velocity * flow_direction,
                        min_coordinates=min_coords_inlet,
                        faces=(false, false, true, true),
                        coordinates_eltype=Float64)

min_coords_outlet = (pipe.fluid_size[1], 0.0)
outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         velocity=prescribed_velocity * flow_direction,
                         min_coordinates=min_coords_outlet,
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

cylinder_center = (5 * cylinder_diameter, domain_size[2] / 2)
cylinder = SphereShape(min_spacing, cylinder_diameter / 2,
                       cylinder_center, fluid_density; sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)

NDIMS = ndims(fluid)

n_buffer_particles = 10 * inlet.n_particles_per_dimension[2]^(NDIMS - 1)

# ==========================================================================================
# ==== Fluid
smoothing_length_ratio = 1.5
smoothing_length = smoothing_length_ratio * particle_spacing
min_smoothing_length = smoothing_length_ratio * min_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

# With `ContinuityDensity`, EDAC has no mechanism that corrects density errors
# (no equation of state, no density diffusion). In the recirculation zone behind the
# cylinder, where particles stay for a long time, the density of single particles drifts
# to more than twice the reference density and the simulation becomes unstable,
# with and without refinement. The summation density is bound to the particle positions.
# This is also the formulation of the flow past a cylinder in the code of Haftu et al.
fluid_density_calculator = SummationDensity()

kinematic_viscosity = prescribed_velocity * cylinder_diameter / reynolds_number
viscosity = ViscosityAdami(nu=kinematic_viscosity)

# ==========================================================================================
# ==== Refinement
spatial_criterion = SpatialRefinementCriterion()

# Refine the wake, where the vorticity exceeds 20 % of the largest vorticity.
# Haftu et al. use 5 %, but at this resolution, the velocity noise of the refined particles
# exceeds 5 % of the largest vorticity, so that the refinement spreads over the domain.
# The vorticity criterion does not refine the last 10 % of the fluid before the outlet.
vorticity_criterion = VorticityRefinementCriterion(; relative_threshold=0.2,
                                                   flow_direction)

refinement_criteria = (spatial_criterion, vorticity_criterion)

n_buffer_refinement_particles = 5000
n_buffer_particles += n_buffer_refinement_particles

refinement_shifting = TrixiParticles.ShiftingHaftu()

# With `spacing_ratio=1.2`, it takes `log(2) / log(1.2) ≈ 4` neighborhoods
# to grade from `min_spacing` to `max_spacing`
refinement = ParticleRefinement(; spacing_ratio=1.2,
                                min_spacing, max_spacing,
                                smoothing_length_factor=smoothing_length_ratio,
                                reference_density=fluid_density,
                                refinement_criteria,
                                shifting_technique=refinement_shifting)

fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                           smoothing_length, sound_speed, viscosity,
                                           density_calculator=fluid_density_calculator,
                                           shifting_technique=ParticleShiftingTechnique(),
                                           buffer_size=n_buffer_particles,
                                           particle_refinement=refinement)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelCharacteristicsLastiwka()

# Prescribe the density in the open boundaries. Extrapolating it from the fluid
# is unstable when particles of different masses enter the boundary zones.
face_in = ([0.0, 0.0], [0.0, domain_size[2]])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density,
                      particle_spacing=particle_spacing,
                      reference_density=fluid_density,
                      reference_velocity=prescribed_velocity * flow_direction,
                      initial_condition=inlet.fluid, boundary_type=InFlow())

face_out = ([min_coords_outlet[1], 0.0], [min_coords_outlet[1], domain_size[2]])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(-flow_direction),
                       open_boundary_layers, density=fluid_density,
                       particle_spacing=particle_spacing,
                       reference_density=fluid_density,
                       reference_pressure=0.0,
                       initial_condition=outlet.fluid, boundary_type=OutFlow())

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
state_equation = nothing

# Free-slip boundary condition for the walls
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)
boundary_model_wall = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length;
                                                  state_equation)
boundary_system_wall = WallBoundarySystem(wall, boundary_model_wall)

# No-slip boundary condition for the cylinder.
# The cylinder has the minimum spacing, so it uses the smoothing length of this spacing.
boundary_model_cylinder = BoundaryModelDummyParticles(cylinder.density, cylinder.mass,
                                                      AdamiPressureExtrapolation(),
                                                      smoothing_kernel,
                                                      min_smoothing_length;
                                                      state_equation, viscosity)
boundary_system_cylinder = WallBoundarySystem(cylinder, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
padding = 4 * smoothing_length
min_corner = minimum(wall.coordinates .- padding, dims=2)
max_corner = maximum(wall.coordinates .+ padding, dims=2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, boundary_system_wall,
                          boundary_system_cylinder;
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="")

extra_callback = nothing

# Builds the graded spacing field before the first time step,
# then refines every 10 time steps
refinement_callback = ParticleRefinementCallback(interval=10)

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback,
                        refinement_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # May need tuning to prevent boundary penetration
            reltol=1e-3, # May need tuning to prevent boundary penetration
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
