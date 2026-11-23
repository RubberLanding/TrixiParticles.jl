# E3: single-cylinder vortex street at Re = 100, convergence of Cd, Cl and St.
#
#   julia --project=run -t 8 experiments/e3_vortex_street/run.jl                  # debug
#   julia --project=run -t 8 experiments/e3_vortex_street/run.jl --profile=gate   # feasibility
#   julia --project=run -t 8 experiments/e3_vortex_street/run.jl --full           # thesis
#
# The example `vortex_street_2d_refinement.jl` with Re = 100 and a larger domain.
# Spacings are given as D/Δx (cylinder diameter over particle spacing).
# Refined runs start the fluid at `max_spacing`; the cylinder has `min_spacing`, walls and
# open boundaries have `max_spacing` (as in the example). Criteria: spatial (near the
# cylinder) and vorticity (wake, 20 % threshold of the example), or spatial alone.
#
# Profiles:
# - `gate`: the feasibility check of the plan.
#   (1) Uniform runs long enough for periodic shedding, with the fluid around the cylinder
#       cut out of the lattice (`cutout`) or as rings (`rings`): Cd and St against the
#       literature, and whether the ~10 % force error of the cutout in the static
#       Archimedes check persists in the developed flow.
#   (2) Short refined runs with resolution ratios 2, 4, 8 at D/Δx_min = 40: stability,
#       density deficit at the cylinder, vorticity noise.
# - `debug`: the whole chain on a small domain, not for interpretation.
# - `full`: study A (D/Δx_min = 20, 40, 80 at a fixed ratio), study B (ratios at
#   D/Δx_min = 40 against uniform fine), spatial vs spatial + vorticity criterion.
#   The resolution ratio and the fluid setup are decided by the gate.

include(joinpath(@__DIR__, "..", "common", "harness.jl"))
include(joinpath(@__DIR__, "cylinder.jl"))

example = joinpath(examples_dir(), "fluid", "vortex_street_2d_refinement.jl")

const D = 0.1
const U = 1.0
const fluid_density = 1000.0
const reynolds_number = 100.0

profile = experiment_profile()

# Domain in multiples of D: length, height and the distance of the cylinder center from
# the inflow. The cylinder is at half height. 20D height gives 5 % blockage.
if profile == "debug"
    domain = (; length=10, height=5, cylinder_x=3)
else
    domain = (; length=30, height=20, cylinder_x=10)
end

# Snapshot grid around the cylinder and the wake, in multiples of D relative to the
# cylinder center, and its resolution
snapshot_window = (; min=(-1.0, -2.0), max=(8.0, 2.0))

if profile == "debug"
    tspan = (0.0, 1.0)
    snapshot_times = [1.0]
    grid_resolution = D / 10
elseif profile == "gate"
    tspan = (0.0, 20.0)
    snapshot_times = [20.0]
    grid_resolution = D / 20
else
    # Shedding period D / (St U) ≈ 0.6. Onset after about 5-10, then ≥ 30 periods.
    tspan = (0.0, 30.0)
    snapshot_times = [30.0]
    grid_resolution = D / 40
end
postprocess_dt = 0.01

# Decided in the gate. The rings remove the force error of the cutout in the static check.
default_fluid_setup = "rings"

function run_cylinder(run_name; resolution, resolution_ratio=1, criteria="vorticity",
                      fluid_setup=default_fluid_setup, tspan=tspan)
    refined = resolution_ratio > 1
    min_spacing = D / resolution
    max_spacing = resolution_ratio * min_spacing

    output_directory = results_directory(@__DIR__, run_name)

    domain_size = (domain.length * D, domain.height * D)
    cylinder_center = (domain.cylinder_x * D, domain_size[2] / 2)

    # Cd/Cl of the cylinder (system 4 in the example: fluid, open boundary, walls, cylinder)
    coefficients = ForceCoefficients(; fluid_index=1, wall_index=4,
                                     reference_density=fluid_density,
                                     reference_velocity=U, reference_length=D)
    snapshot_times_ = filter(t -> t <= tspan[2] + 1e-8, snapshot_times)
    isempty(snapshot_times_) && (snapshot_times_ = [tspan[2]])
    grid_min = cylinder_center .+ D .* snapshot_window.min
    grid_max = cylinder_center .+ D .* snapshot_window.max
    quantities_ = merge(quantities(coefficients),
                        snapshot_quantity(output_directory, snapshot_times_;
                                          grid_min, grid_max, grid_resolution))

    overrides = (; tspan, reynolds_number, domain_size, cylinder_center,
                 particle_spacing=max_spacing, min_spacing,
                 info_callback=InfoCallback(interval=profile == "debug" ? 100 : 1000))

    if fluid_setup == "rings"
        # The fluid tank of the example, with rings around the cylinder
        tank = RectangularTank(max_spacing, domain_size, domain_size, fluid_density;
                               n_layers=4, velocity=TP.SVector(U, 0.0),
                               faces=(false, false, true, true),
                               coordinates_eltype=Float64)
        fluid = ring_fluid(tank.fluid, max_spacing, D / 2, cylinder_center, fluid_density;
                           ring_thickness=D / 2, velocity=TP.SVector(U, 0.0))
        overrides = merge(overrides, (; fluid))
    end

    if refined
        # Enough for half of the domain at `min_spacing`
        n_buffer_refinement_particles = ceil(Int,
                                             0.5 * prod(domain_size) / min_spacing^2)
        overrides = merge(overrides, (; n_buffer_refinement_particles))

        if criteria == "spatial"
            overrides = merge(overrides,
                              (; refinement_criteria=(SpatialRefinementCriterion(),)))
        end
    else
        overrides = merge(overrides, (; refinement=nothing, refinement_callback=nothing,
                                      n_buffer_refinement_particles=0))
    end

    parameters = Dict("resolution" => resolution, "min_spacing" => min_spacing,
                      "max_spacing" => max_spacing, "resolution_ratio" => resolution_ratio,
                      "refined" => refined, "criteria" => refined ? criteria : "none",
                      "fluid_setup" => fluid_setup, "reynolds_number" => reynolds_number,
                      "diameter" => D, "domain" => Dict(pairs(domain)),
                      "cylinder_center" => collect(cylinder_center))

    run_case(@__DIR__, run_name, example; overrides, quantities=quantities_,
             postprocess_dt, save_times=snapshot_times_, parameters)
end

run_name(resolution, ratio, criteria, fluid_setup) =
    ratio == 1 ? "uniform_D$(resolution)_$(fluid_setup)" :
    "refined_D$(resolution)_ratio$(ratio)_$(criteria)_$(fluid_setup)"

if profile == "debug"
    for resolution in (5, 10)
        run_cylinder(run_name(resolution, 1, "", "rings"); resolution)
        run_cylinder(run_name(resolution, 2, "vorticity", "rings"); resolution,
                     resolution_ratio=2)
    end
    run_cylinder(run_name(10, 2, "spatial", "rings"); resolution=10, resolution_ratio=2,
                 criteria="spatial")
    run_cylinder(run_name(10, 1, "", "cutout"); resolution=10, fluid_setup="cutout")
elseif profile == "gate"
    # (1) Force validation on uniform runs
    for resolution in (10, 20), fluid_setup in ("cutout", "rings")
        run_cylinder(run_name(resolution, 1, "", fluid_setup); resolution, fluid_setup)
    end

    # (2) Short refined runs
    for ratio in (2, 4, 8)
        run_cylinder(run_name(40, ratio, "vorticity", "rings"); resolution=40,
                     resolution_ratio=ratio, tspan=(0.0, 3.0))
    end
else
    # The fixed resolution ratio of study A and the ratios of study B, after the gate
    ratio_a = 2
    ratios_b = (2, 4, 8)

    # Uniform series
    for resolution in (20, 40, 80)
        run_cylinder(run_name(resolution, 1, "", default_fluid_setup); resolution)
    end

    # Study A
    for resolution in (20, 40, 80)
        run_cylinder(run_name(resolution, ratio_a, "vorticity", default_fluid_setup);
                     resolution, resolution_ratio=ratio_a)
    end

    # Study B and the criteria comparison at D/Δx_min = 40
    for ratio in ratios_b
        run_cylinder(run_name(40, ratio, "vorticity", default_fluid_setup);
                     resolution=40, resolution_ratio=ratio)
    end
    run_cylinder(run_name(40, ratio_a, "spatial", default_fluid_setup); resolution=40,
                 resolution_ratio=ratio_a, criteria="spatial")
end
