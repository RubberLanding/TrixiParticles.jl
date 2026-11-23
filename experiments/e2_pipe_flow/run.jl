# E2: pipe flow (plane Poiseuille flow), convergence with walls and open boundaries.
#
#   julia --project=run -t 8 experiments/e2_pipe_flow/run.jl            # debug profile
#   julia --project=run -t 8 experiments/e2_pipe_flow/run.jl --full     # thesis results
#   ... --full --only=uniform                                              # a part of the sweep
#
# The example `pipe_flow_2d_refinement.jl` with a parabolic inflow and initial velocity
# (see `poiseuille.jl`), mean velocity 1 and Re = U_mean H / ν = 100.
# Runs (study A of the plan):
# - uniform series: `particle_spacing` = Δx without refinement, walls and open boundaries
#   at Δx.
# - refined series (resolution ratio 2): `min_spacing` = Δx, `max_spacing` = 2Δx.
#   Walls and open boundaries are at `min_spacing`. The example refines the fluid near the
#   walls and open boundaries (spatial criterion) and in the boxes x < 0.15 and x > 0.85,
#   so the channel core between the boxes is coarse.

include(joinpath(@__DIR__, "..", "common", "harness.jl"))
include(joinpath(@__DIR__, "poiseuille.jl"))

example = joinpath(examples_dir(), "fluid", "pipe_flow_2d_refinement.jl")

if is_full_profile()
    levels = [0.02, 0.01, 0.005, 0.0025]
    tspan = (0.0, 3.0)
    postprocess_dt = 0.05
    snapshot_times = [2.0, 2.25, 2.5, 2.75, 3.0]
    n_profile_points = 80
else
    levels = [0.04, 0.02]
    tspan = (0.0, 0.2)
    postprocess_dt = 0.02
    snapshot_times = [0.1, 0.2]
    n_profile_points = 40
end

const reynolds_number = 100.0
const fluid_density = 1000.0
const U_mean = 1.0
const channel_height = 0.4
const channel_length = 1.0
const kinematic_viscosity = U_mean * channel_height / reynolds_number

flow = PoiseuilleFlow(; H=channel_height, L=channel_length, U_mean, kinematic_viscosity,
                      density=fluid_density, region=(0.2, 0.8))

# Ma = 0.1 with the largest velocity
sound_speed = 10 * flow.U_max

# x positions of the profiles, in the coarse core of the refined runs
stations = [0.25, 0.5, 0.75]

inflow_velocity(x, _) = exact_velocity(flow, x)

# The tank `pipe` of the example with the exact velocity and pressure as initial condition,
# so that the run starts close to the steady state. With the pressure 0 of the example,
# the pressure gradient builds up in acoustic waves that take long to decay.
function pipe_tank(max_spacing, resolution_ratio)
    domain_size = (channel_length, channel_height)
    tank = RectangularTank(max_spacing, domain_size, domain_size, fluid_density;
                           n_layers=4, velocity=x -> exact_velocity(flow, x),
                           faces=(false, false, true, true), coordinates_eltype=Float64,
                           spacing_ratio=resolution_ratio)

    for particle in axes(tank.fluid.coordinates, 2)
        tank.fluid.pressure[particle] = exact_pressure(flow,
                                                       tank.fluid.coordinates[:, particle])
    end

    return tank
end

function run_pipe(run_name; min_spacing, refined)
    output_directory = results_directory(@__DIR__, run_name)
    quantities = merge(poiseuille_quantities(flow),
                       profile_snapshot_quantity(flow, output_directory, snapshot_times;
                                                 stations, n_points=n_profile_points,
                                                 n_points_centerline=2 * n_profile_points))

    resolution_ratio = refined ? 2 : 1
    max_spacing = resolution_ratio * min_spacing

    # Unrefined runs have no refinement buffer. Refined runs: the whole channel at
    # `min_spacing` is always enough.
    n_buffer_refinement_particles = refined ?
                                    ceil(Int, channel_height * channel_length /
                                              min_spacing^2) : 0

    overrides = (; tspan, particle_spacing=max_spacing,
                 pipe=pipe_tank(max_spacing, resolution_ratio), sound_speed,
                 kinematic_viscosity,
                 reference_velocity_in=inflow_velocity, n_buffer_refinement_particles,
                 info_callback=InfoCallback(interval=is_full_profile() ? 1000 : 50))

    # Refined runs keep the refinement and the resolution ratio 2 of the example.
    # Note that `trixi_include` also replaces keyword arguments: overriding
    # `spacing_ratio` (the example's name for the resolution ratio) also replaces the
    # grading `spacing_ratio=1.2` in `ParticleRefinement(...)`. This is only safe for
    # the uniform runs, where `refinement` is replaced as a whole.
    if !refined
        overrides = merge(overrides, (; spacing_ratio=1, refinement=nothing,
                                      refinement_callback=nothing))
    end

    parameters = Dict("min_spacing" => min_spacing, "max_spacing" => max_spacing,
                      "refined" => refined, "resolution_ratio" => resolution_ratio,
                      "sound_speed" => sound_speed, "reynolds_number" => reynolds_number,
                      "U_max" => flow.U_max, "dpdx_exact" => -flow.G,
                      "region" => collect(flow.region), "stations" => stations)

    run_case(@__DIR__, run_name, example; overrides, quantities, postprocess_dt,
             parameters)
end

for dx in levels
    run_pipe("uniform_dx$(dx)"; min_spacing=dx, refined=false)
end

for dx in levels
    run_pipe("refined_max$(2 * dx)_min$(dx)"; min_spacing=dx, refined=true)
end
