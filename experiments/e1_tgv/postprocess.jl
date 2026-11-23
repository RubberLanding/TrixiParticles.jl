# E1: postprocessing of the Taylor-Green vortex runs.
#
#   julia --project=run experiments/e1_tgv/postprocess.jl [--full]
#
# Writes to `figure_data_directory("e1_tgv")`:
#   convergence_uniform.csv     errors at `t_eval` and observed orders, uniform series
#   convergence_refined.csv     the same for study A (refined, resolution ratio 2); both
#                               tables have `dx` (= `min_spacing`) and the measured
#                               `dx_effective`, with the orders over both
#   runs.csv                    all runs with cost and errors (accuracy vs cost, study B)
#   kinetic_energy_<run>.csv    kinetic energy and exact kinetic energy over time
#   error_map_uniform.png       |u - u_exact| on the grid at `t_eval` (E1c) for the
#   error_map_refined.png       finest uniform run and the refined run with the same
#                               `min_spacing` (study A), and
#   error_map_clims.csv         the color limits of these images

include(joinpath(@__DIR__, "..", "common", "postprocess_util.jl"))

runs = load_runs(@__DIR__)
output = figure_data_directory("e1_tgv")

# Time at which the convergence is evaluated
t_eval = is_full_profile() ? 1.0 : 0.2

error_columns = [:v_l1, :v_l2, :p_l1, :p_l2, :v_l1_box, :v_l2_box, :p_l1_box, :p_l2_box]

# ==========================================================================================
# ==== Table of all runs
mean_particles(metrics) = sum(metrics.n_particles) / nrow(metrics)

rows = map(runs) do run
    p = run.parameters
    m = metrics_at(run.metrics, t_eval)

    (; name=run.name, refined=p["refined"], min_spacing=p["min_spacing"],
     max_spacing=p["max_spacing"], resolution_ratio=p["resolution_ratio"],
     dx_effective=effective_spacing(run), n_particles_mean=mean_particles(run.metrics),
     n_particles_max=maximum(run.metrics.n_particles),
     wall_time_total=run.summary["wall_time"],
     wall_time_stepping=run.metrics.wall_time[end],
     n_steps=run.summary["n_steps_accepted"], max_rss_mb=run.summary["max_rss_mb"],
     n_threads=run.summary["n_threads"],
     (column => m[column] for column in error_columns)...)
end

table = DataFrame(rows)
sort!(table, [:refined, :max_spacing, :min_spacing], rev=[false, true, true])
CSV.write(joinpath(output, "runs.csv"), table)

# ==========================================================================================
# ==== Convergence tables (study A)
function convergence_table(rows)
    rows = sort(rows, :min_spacing, rev=true)
    # `dx` is the `min_spacing` parameter, `dx_effective` the measured smallest spacing
    # (see `effective_spacing`), and the orders are computed over both
    result = DataFrame(dx=rows.min_spacing, dx_effective=rows.dx_effective)
    for column in error_columns
        result[!, column] = rows[!, column]
        result[!, Symbol(column, :_order)] = observed_order(rows.min_spacing, rows[!, column])
        result[!, Symbol(column, :_order_effective)] = observed_order(rows.dx_effective,
                                                                      rows[!, column])
    end

    return result
end

uniform = table[.!table.refined, :]
refined_a = table[table.refined .& (table.resolution_ratio .≈ 2), :]

CSV.write(joinpath(output, "convergence_uniform.csv"), convergence_table(uniform))
CSV.write(joinpath(output, "convergence_refined.csv"), convergence_table(refined_a))

# ==========================================================================================
# ==== Kinetic energy
for run in runs
    ke = DataFrame(time=run.metrics.time, kinetic_energy=run.metrics.kinetic_energy,
                   kinetic_energy_exact=run.metrics.kinetic_energy_exact)
    ke.relative_error = ke.kinetic_energy ./ ke.kinetic_energy_exact .- 1
    CSV.write(joinpath(output, "kinetic_energy_$(run.name).csv"), ke)
end

# ==========================================================================================
# ==== Error maps (E1c): uniform fine and refined run with the same `min_spacing`
grid_file(run) = joinpath(run.directory, "grid_t$(round(t_eval, digits=4)).csv")

function velocity_error(grid)
    return sqrt.((grid.vx .- grid.vx_exact) .^ 2 .+ (grid.vy .- grid.vy_exact) .^ 2)
end

# The finest level that has both a uniform run and a refined run of study A
finest = minimum(intersect(uniform.min_spacing, refined_a.min_spacing))
map_runs = filter(run -> run.parameters["min_spacing"] == finest &&
                         (!run.parameters["refined"] ||
                          run.parameters["resolution_ratio"] ≈ 2), runs)

grids = Dict(run.name => CSV.read(grid_file(run), DataFrame) for run in map_runs)

# Shared color limits, so that the images can be compared
clims = (0.0, maximum(maximum(velocity_error(grid)) for grid in values(grids)))

# Fixed file names `error_map_uniform.png` and `error_map_refined.png`, so that the figure
# works with the debug and the full data
for run in map_runs
    grid = grids[run.name]
    kind = run.parameters["refined"] ? "refined" : "uniform"
    write_image(joinpath(output, "error_map_$kind.png"), grid.x, grid.y,
                velocity_error(grid); clims)
end

CSV.write(joinpath(output, "error_map_clims.csv"),
          DataFrame(runs=join([run.name for run in map_runs], " "), t=t_eval,
                    min_spacing=finest, min=clims[1], max=clims[2]))

println("E1 postprocessing ($(experiment_profile())) written to ", abspath(output))
println(convergence_table(uniform)[:, [:dx, :v_l2, :v_l2_order, :p_l2, :p_l2_order]])
println(convergence_table(refined_a)[:, [:dx, :v_l2, :v_l2_order, :v_l2_box, :v_l2_box_order]])
