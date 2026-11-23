# E1: Taylor-Green vortex, convergence against the exact solution.
#
#   julia --project=run -t 8 experiments/e1_tgv/run.jl            # debug profile
#   julia --project=run -t 8 experiments/e1_tgv/run.jl --full     # thesis results
#   ... --full --only=uniform                                        # a part of the sweep
#
# Runs (see the plan):
# - uniform series: `particle_spacing` = Δx without refinement.
# - study A (fixed resolution ratio 2): refined runs with `min_spacing` = Δx and
#   `max_spacing` = 2Δx, for the same Δx as the uniform series.
# - study B (fixed `max_spacing`): refined runs with resolution ratios 2, 4, 8.
# Refined runs start at `max_spacing`, and the box criterion of the example refines the
# box [0.35, 0.65]² to `min_spacing`.

include(joinpath(@__DIR__, "..", "common", "harness.jl"))
include(joinpath(@__DIR__, "tgv_quantities.jl"))

example = joinpath(examples_dir(), "fluid", "taylor_green_vortex_2d_refinement.jl")

if is_full_profile()
    levels = [0.05, 0.025, 0.0125, 0.00625]
    study_b_max_spacing = 0.05
    study_b_ratios = [2, 4, 8]
    tspan = (0.0, 2.0)
    postprocess_dt = 0.02
    snapshot_times = [1.0, 2.0]
    snapshot_resolution = 0.005
else
    levels = [0.05, 0.025]
    study_b_max_spacing = 0.1
    study_b_ratios = [2, 4]
    tspan = (0.0, 0.2)
    postprocess_dt = 0.02
    snapshot_times = [0.2]
    snapshot_resolution = 0.02
end

# Grading between neighboring levels (C_r in Haftu et al.). The example uses 1.05.
# With 1.05, grading from `min_spacing` to `max_spacing` = 8 `min_spacing` takes about
# log(8) / log(1.05) ≈ 43 neighborhoods, which is more than the periodic box, so study B
# would never reach the coarse spacing. 1.2 is the value of the other refinement examples.
refinement_spacing_ratio = 1.2

const reynolds_number = 100.0
const sound_speed = 10.0
const U = 1.0
tgv = TaylorGreenVortex(; U, reynolds_number)
nu = U / reynolds_number

# The smoothing length factor of the example (with the quintic spline kernel)
smoothing_length_factor = 1.0

# The time step limit (Eq. 31 in Haftu et al.) from the smallest smoothing length.
# Note that the example writes `smoothing_length / 4 * (sound_speed + U)`, which is
# h (c + U) / 4 instead of h / (4 (c + U)), so its CFL limit is never active there.
function dt_max(min_spacing)
    h = smoothing_length_factor * min_spacing
    return min(h / (4 * (sound_speed + U)), h^2 / (8 * nu))
end

# The refinement of the example with the grading `refinement_spacing_ratio`
function refinement(min_spacing, max_spacing)
    box_logic = (system, particle, v, u, semi) -> begin
        position = TrixiParticles.current_coords(u, system, particle)
        return in_box(tgv, position) ? min_spacing : max_spacing
    end

    return ParticleRefinement(; spacing_ratio=refinement_spacing_ratio,
                              smoothing_length_factor, min_spacing, max_spacing,
                              reference_density=1.0,
                              refinement_criteria=(SolutionRefinementCriterion(box_logic),))
end

function run_tgv(run_name; particle_spacing, min_spacing, refined)
    output_directory = results_directory(@__DIR__, run_name)
    quantities = merge(tgv_quantities(tgv),
                       grid_snapshot_quantity(tgv, output_directory, snapshot_times;
                                              resolution=snapshot_resolution))

    overrides = (; tspan, particle_spacing, dt_max=dt_max(min_spacing),
                 refinement=refined ? refinement(min_spacing, particle_spacing) : nothing,
                 refinement_callback=refined ? ParticleRefinementCallback(interval=10) :
                                     nothing,
                 # The whole box at `min_spacing`, twice, is always enough
                 n_buffer_particles=refined ? ceil(Int, 2 / min_spacing^2) : nothing,
                 info_callback=InfoCallback(interval=is_full_profile() ? 1000 : 50))

    parameters = Dict("particle_spacing" => particle_spacing, "min_spacing" => min_spacing,
                      "max_spacing" => particle_spacing, "refined" => refined,
                      "resolution_ratio" => particle_spacing / min_spacing,
                      "refinement_spacing_ratio" => refinement_spacing_ratio)

    run_case(@__DIR__, run_name, example; overrides, quantities, postprocess_dt,
             callback_variable=:pp_callback, parameters)
end

# Uniform series
for dx in levels
    run_tgv("uniform_dx$(dx)"; particle_spacing=dx, min_spacing=dx, refined=false)
end

# Refined runs of study A and study B. Some pairs appear in both studies, so collect
# the unique pairs (max_spacing, min_spacing).
pairs_a = [(2 * dx, dx) for dx in levels]
pairs_b = [(study_b_max_spacing, study_b_max_spacing / ratio) for ratio in study_b_ratios]

for (max_spacing, min_spacing) in unique(vcat(pairs_a, pairs_b))
    run_tgv("refined_max$(max_spacing)_min$(min_spacing)";
            particle_spacing=max_spacing, min_spacing, refined=true)
end
