# Run harness for the thesis experiments.
#
# Each experiment folder has a `run.jl` that calls `run_case` once per run of its sweep.
# `run_case` includes a TrixiParticles example with overrides in a fresh module,
# adds the metrics (and force) callback, and writes all results of the run to
#   <experiment>/results/<profile>/<run name>/
#     metrics.csv    time series from the `PostprocessCallback`
#     summary.json   wall time, steps, memory, final particle count, parameters
#     timer.json     TrixiParticles timer (also as timer.txt)
#     vtk/           particle snapshots (only if `save_dt` or `save_times` is given)
#
# Profiles: `debug` (default) or `full`, chosen by the environment variable
# `EXPERIMENT_PROFILE` or the command line argument `--full`:
#   julia --project=run -t 8 experiments/e1_tgv/run.jl --full
# Debug results are never used in the thesis (see `postprocess.jl` of each experiment).
# Experiments can define further profiles, chosen with `--profile=<name>`
# (e.g. the feasibility check of E3, `--profile=gate`).

using TrixiParticles

include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "forces.jl"))

function experiment_profile()
    for arg in ARGS
        startswith(arg, "--profile=") && return arg[(length("--profile=") + 1):end]
    end

    if "--full" in ARGS
        return "full"
    end

    return get(ENV, "EXPERIMENT_PROFILE", "debug")
end

is_full_profile() = experiment_profile() == "full"

# `--only=<pattern>` runs only the runs whose name contains `<pattern>`, so that a sweep
# can be split into several processes, e.g. `--only=uniform` and `--only=refined`.
function is_selected(run_name)
    patterns = [arg[(length("--only=") + 1):end]
                for arg in ARGS if startswith(arg, "--only=")]

    return isempty(patterns) || any(pattern -> occursin(pattern, run_name), patterns)
end

# Resolution ratios (max_spacing / min_spacing) above this give near-coincident particles
# after the refinement startup, sometimes NaN, and a collapsing time step (see the plan,
# "E1 status"). Sweeps keep their planned ratios and run only the ones up to this limit
# (user, 2026-11-22). Raise it once the startup is fixed.
const MAX_STABLE_RESOLUTION_RATIO = 2

function stable_resolution_ratios(ratios)
    stable = filter(ratio -> ratio <= MAX_STABLE_RESOLUTION_RATIO, collect(ratios))
    skipped = setdiff(collect(ratios), stable)
    isempty(skipped) ||
        @warn "Skipping resolution ratios $skipped (> MAX_STABLE_RESOLUTION_RATIO = $MAX_STABLE_RESOLUTION_RATIO)"

    return stable
end

function results_directory(experiment_directory, run_name)
    return joinpath(experiment_directory, "results", experiment_profile(), run_name)
end

"""
    run_case(experiment_directory, run_name, example; overrides=(;), fluid_index=1,
             quantities=(;), postprocess_dt, save_dt=nothing, save_times=nothing,
             callback_variable=:extra_callback, warmup=is_full_profile(),
             skip_existing=true, parameters=Dict())

Run `example` (a path) with `overrides` passed to `trixi_include` and write the results to
`results_directory(experiment_directory, run_name)`.

- `quantities`: extra custom quantities for the `PostprocessCallback`, e.g. the force
  coefficients from `quantities(ForceCoefficients(...))`.
- `callback_variable`: the variable of the example that takes an extra callback
  (`extra_callback` or `pp_callback`).
- `warmup`: run the example for a few time steps first, so that the wall time of the run
  does not include compilation. On by default in the `full` profile.
- `skip_existing`: skip runs that already have a `summary.json`, so that an interrupted
  sweep can be restarted.
- `parameters`: written to `summary.json` to identify the run in the postprocessing.
- `save_dt`, `save_times`: VTK output of all systems every `save_dt` or at `save_times`.

Returns the summary `Dict`, or `nothing` if the run is not selected with `--only`.
"""
function run_case(experiment_directory, run_name, example; overrides=(;), fluid_index=1,
                  quantities=(;), postprocess_dt, save_dt=nothing, save_times=nothing,
                  callback_variable=:extra_callback, warmup=is_full_profile(),
                  skip_existing=true, parameters=Dict())
    output_directory = results_directory(experiment_directory, run_name)
    summary_file = joinpath(output_directory, "summary.json")

    is_selected(run_name) || return nothing

    if skip_existing && isfile(summary_file)
        @info "Skipping $run_name, results exist" output_directory
        return JSON.parsefile(summary_file)
    end

    if warmup
        @info "Warm-up run for $run_name"
        # Only `tspan` and the output differ from the run
        trixi_include(Module(), example;
                      merge(overrides, (; tspan=(0.0, 1e-4), saving_callback=nothing,
                                        info_callback=nothing))...)
    end

    postprocess_callback = PostprocessCallback(; dt=postprocess_dt, output_directory,
                                               filename="metrics", write_json=false,
                                               run_metrics(; fluid_index)...,
                                               quantities...)

    saving_callback = if isnothing(save_dt) && isnothing(save_times)
        nothing
    else
        SolutionSavingCallback(; dt=something(save_dt, 0.0),
                               save_times=something(save_times, Float64[]), prefix="",
                               save_initial_solution=false,
                               output_directory=joinpath(output_directory, "vtk"))
    end

    run_overrides = merge(overrides, (; saving_callback),
                          NamedTuple{(callback_variable,)}((postprocess_callback,)))

    @info "Running $run_name ($(experiment_profile()))" output_directory
    TP.reset_timer!(TP.timer())

    # A fresh module per run, since examples define constants
    mod = Module()
    wall_time = @elapsed trixi_include(mod, example; run_overrides...)

    fluid_system = mod.semi.systems[fluid_index]
    parameters_ = merge(Dict{String, Any}("run_name" => run_name,
                                          "profile" => experiment_profile(),
                                          "example" => basename(example)),
                        Dict{String, Any}(string(k) => v for (k, v) in parameters))

    return write_run_summary(output_directory, mod.sol, mod.semi, fluid_system;
                             wall_time, parameters=parameters_)
end
