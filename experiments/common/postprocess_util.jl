# Shared helpers for the `postprocess.jl` scripts of the experiments.
#
# The postprocessing reads `results/<profile>/<run>/` and writes small CSV files and PNG
# images to the thesis data folder, from which the pgfplots figures are drawn:
#   ../thesis/figures/data/<experiment>/          (full profile)
#   ../thesis/figures/data/debug/<experiment>/    (debug profile)

using CSV
using DataFrames
using JSON
using Plots

include(joinpath(@__DIR__, "harness.jl"))

# The thesis is next to the TrixiParticles worktree that contains this folder
const thesis_figure_data = joinpath(@__DIR__, "..", "..", "..", "thesis", "figures", "data")

function figure_data_directory(experiment)
    directory = is_full_profile() ? joinpath(thesis_figure_data, experiment) :
                joinpath(thesis_figure_data, "debug", experiment)
    mkpath(directory)

    return directory
end

"""
    load_runs(experiment_directory)

All finished runs of the current profile as a vector of
`(; name, directory, parameters, summary, metrics)`, where `metrics` is a `DataFrame`
whose columns have the `_fluid_1` suffix of the `PostprocessCallback` removed.
"""
function load_runs(experiment_directory)
    directory = joinpath(experiment_directory, "results", experiment_profile())
    isdir(directory) || error("no results in $directory, run `run.jl` first")

    runs = []
    for name in sort(readdir(directory))
        run_directory = joinpath(directory, name)
        summary_file = joinpath(run_directory, "summary.json")
        isfile(summary_file) || continue

        summary = JSON.parsefile(summary_file)
        metrics = CSV.read(joinpath(run_directory, "metrics.csv"), DataFrame)
        rename!(column -> replace(column, r"_fluid_\d+$" => ""), metrics)

        push!(runs, (; name, directory=run_directory, parameters=summary["parameters"],
                     summary, metrics))
    end

    return runs
end

# The row of `metrics` at time `t` (the `PostprocessCallback` writes at multiples of its `dt`)
function metrics_at(metrics, t)
    index = findfirst(time -> isapprox(time, t; atol=1e-8), metrics.time)
    isnothing(index) && error("no metrics at t = $t")

    return metrics[index, :]
end

"""
    effective_spacing(run)

The measured smallest particle spacing of a run: the smallest smoothing length (median
over the samples after `t = 0`) divided by the smoothing length factor h/Δx of the
initial particles, which all have `max_spacing`.

For a uniform run, this is the particle spacing. For a refined run, it is the spacing of
the finest particles that the refinement actually makes, which can differ from the
`min_spacing` parameter: at resolution ratio 2, the hexagonal split gives 7 daughters of
mass m/7 that never merge back, i.e. a spacing of Δx_max/√7 ≈ 0.76 Δx_min.
`h_min` is an extreme over all particles, so local density changes enter as well.
"""
function effective_spacing(run)
    h_factor = run.metrics.h_max[1] / run.parameters["max_spacing"]
    after_start = run.metrics.time .> 0
    h_min = any(after_start) ? run.metrics.h_min[after_start] : run.metrics.h_min
    sorted = sort(h_min)

    return sorted[(length(sorted) + 1) ÷ 2] / h_factor
end

# Observed order of convergence between consecutive levels: log(e_i / e_{i+1}) / log(Δx_i / Δx_{i+1}).
# The first level has no order (`NaN`).
function observed_order(spacing, error)
    order = fill(NaN, length(spacing))
    for i in 2:length(spacing)
        order[i] = log(error[i - 1] / error[i]) / log(spacing[i - 1] / spacing[i])
    end

    return order
end

"""
    write_image(filename, x, y, values; clims, colormap=:viridis, pixels=800)

Write a field on a regular grid as a PNG without axes, labels or colorbar, filling the
image completely. pgfplots draws the axes with `\\addplot graphics` around it, with the
same `clims` (`point meta min/max`) and colormap.
`x` and `y` are the coordinates of the points, `values` the field at the points.
"""
function write_image(filename, x, y, values; clims, colormap=:viridis, pixels=800)
    xs = sort(unique(x))
    ys = sort(unique(y))
    grid = fill(NaN, length(ys), length(xs))
    index_x = Dict(value => i for (i, value) in enumerate(xs))
    index_y = Dict(value => i for (i, value) in enumerate(ys))
    for (xi, yi, value) in zip(x, y, values)
        grid[index_y[yi], index_x[xi]] = value
    end

    aspect = (xs[end] - xs[1]) / (ys[end] - ys[1])
    plot_ = heatmap(xs, ys, grid; clims, color=colormap, colorbar=false, legend=false,
                    axis=false, ticks=false, grid=false, framestyle=:none,
                    size=(round(Int, pixels * aspect), pixels), margin=0Plots.mm,
                    background_color=:transparent)
    savefig(plot_, filename)

    return filename
end
