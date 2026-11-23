# E2: postprocessing of the pipe flow runs.
#
#   julia --project=run experiments/e2_pipe_flow/postprocess.jl [--full]
#
# All quantities are averaged over the steady-state window, i.e., over the snapshot times
# of `run.jl` (the last outputs of each run).
#
# Writes to `figure_data_directory("e2_pipe_flow")`:
#   runs.csv                  all runs with cost, errors and the flux at the stations
#   convergence_uniform.csv   errors and observed orders, uniform series
#   convergence_refined.csv   the same for the refined series (resolution ratio 2)
#   profile_uniform.csv       u(y) at the stations (columns `vx_<station>`) with the
#   profile_refined.csv       exact profile, for the finest uniform and refined run (E2a)
#   centerline_uniform.csv    p(x) on the centerline with the exact pressure, for the same
#   centerline_refined.csv    runs
#   history_<run>.csv         errors, pressure gradient, particle count and mass over time
#
# Errors:
# - `profile_l1`, `profile_l2`: relative L1/L2 error of the interpolated u(y) at all
#   stations against the Poiseuille profile (the convergence quantity of the plan).
# - `v_l1`, `v_l2`: the same over the particles in the measurement region.
# - `dpdx_error`: relative error of the fitted pressure gradient.

include(joinpath(@__DIR__, "..", "common", "postprocess_util.jl"))

runs = load_runs(@__DIR__)
output = figure_data_directory("e2_pipe_flow")

snapshot_suffix(t) = "t$(round(t, digits=4)).csv"

# Snapshot times are the times with a `profiles_*.csv` file
function snapshot_times(run)
    files = filter(file -> startswith(file, "profiles_t"), readdir(run.directory))
    times = [parse(Float64, file[(length("profiles_t") + 1):(end - 4)]) for file in files]

    return sort(times)
end

# Mean of the snapshot tables over the snapshot times (all tables have the same points)
function mean_snapshot(run, prefix)
    tables = [CSV.read(joinpath(run.directory, "$(prefix)_" * snapshot_suffix(t)),
                       DataFrame) for t in snapshot_times(run)]
    result = copy(tables[1])
    for column in names(result)
        column in ("x", "y") && continue
        result[!, column] = sum(table[!, column] for table in tables) / length(tables)
    end

    return result
end

mean_over(metrics, column, times) = sum(metrics_at(metrics, t)[column] for t in times) /
                                    length(times)

# ==========================================================================================
# ==== Table of all runs
rows = map(runs) do run
    p = run.parameters
    times = snapshot_times(run)
    profiles = mean_snapshot(run, "profiles")

    if any(isnan, profiles.vx)
        @warn "NaN in the interpolated profiles of $(run.name)"
    end

    difference = profiles.vx .- profiles.vx_exact
    profile_l1 = sum(abs, difference) / sum(abs, profiles.vx_exact)
    profile_l2 = sqrt(sum(abs2, difference) / sum(abs2, profiles.vx_exact))

    # Volume flux ∫ u dy at each station (midpoint rule on the cell-centered points),
    # relative to the exact flux U_mean H
    dy = profiles.y[2] - profiles.y[1]
    flux_exact = sum(profiles.vx_exact[profiles.x .== p["stations"][1]]) * dy
    flux = [sum(profiles.vx[profiles.x .== station]) * dy / flux_exact - 1
            for station in p["stations"]]

    (; name=run.name, refined=p["refined"], min_spacing=p["min_spacing"],
     max_spacing=p["max_spacing"], resolution_ratio=p["resolution_ratio"],
     dx_effective=effective_spacing(run), n_particles_mean=sum(run.metrics.n_particles) / nrow(run.metrics),
     n_particles_max=maximum(run.metrics.n_particles),
     wall_time_total=run.summary["wall_time"],
     wall_time_stepping=run.metrics.wall_time[end],
     n_steps=run.summary["n_steps_accepted"], max_rss_mb=run.summary["max_rss_mb"],
     n_threads=run.summary["n_threads"], t_average_start=times[1],
     t_average_end=times[end], profile_l1, profile_l2,
     v_l1=mean_over(run.metrics, :v_l1, times),
     v_l2=mean_over(run.metrics, :v_l2, times),
     dpdx=mean_over(run.metrics, :dpdx, times), dpdx_exact=p["dpdx_exact"],
     dpdx_error=mean_over(run.metrics, :dpdx, times) / p["dpdx_exact"] - 1,
     (Symbol("flux_error_x", station) => flux[i]
      for (i, station) in enumerate(p["stations"]))...)
end

table = DataFrame(rows)
sort!(table, [:refined, :min_spacing], rev=[false, true])
CSV.write(joinpath(output, "runs.csv"), table)

# ==========================================================================================
# ==== Convergence tables
error_columns = [:profile_l1, :profile_l2, :v_l1, :v_l2, :dpdx_error]

function convergence_table(rows)
    rows = sort(rows, :min_spacing, rev=true)
    # `dx` is the `min_spacing` parameter, `dx_effective` the measured smallest spacing
    # (see `effective_spacing`), and the orders are computed over both
    result = DataFrame(dx=rows.min_spacing, dx_effective=rows.dx_effective)
    for column in error_columns
        # The pressure gradient error can have either sign
        values = abs.(rows[!, column])
        result[!, column] = values
        result[!, Symbol(column, :_order)] = observed_order(rows.min_spacing, values)
        result[!, Symbol(column, :_order_effective)] = observed_order(rows.dx_effective,
                                                                      values)
    end

    return result
end

uniform = table[.!table.refined, :]
refined = table[table.refined, :]

CSV.write(joinpath(output, "convergence_uniform.csv"), convergence_table(uniform))
CSV.write(joinpath(output, "convergence_refined.csv"), convergence_table(refined))

# ==========================================================================================
# ==== Profiles and centerline pressure (E2a) of the finest uniform and refined run
# The finest level that has both a uniform and a refined run
finest = minimum(intersect(uniform.min_spacing, refined.min_spacing))

for run in runs
    run.parameters["min_spacing"] == finest || continue
    kind = run.parameters["refined"] ? "refined" : "uniform"

    profiles = mean_snapshot(run, "profiles")
    stations = run.parameters["stations"]
    at_station(station) = profiles[profiles.x .== station, :]

    wide = DataFrame(y=at_station(stations[1]).y,
                     vx_exact=at_station(stations[1]).vx_exact)
    for station in stations
        wide[!, "vx_$station"] = at_station(station).vx
    end
    CSV.write(joinpath(output, "profile_$kind.csv"), wide)

    CSV.write(joinpath(output, "centerline_$kind.csv"), mean_snapshot(run, "centerline"))
end

# ==========================================================================================
# ==== Time histories (steady state of errors, pressure gradient and particle count)
for run in runs
    history = select(run.metrics, :time, :v_l1, :v_l2, :dpdx, :dpdx_exact, :n_particles,
                     :total_mass, :momentum_x)
    CSV.write(joinpath(output, "history_$(run.name).csv"), history)
end

println("E2 postprocessing ($(experiment_profile())) written to ", abspath(output))
println(convergence_table(uniform)[:, [:dx, :profile_l2, :profile_l2_order, :v_l2,
                                      :v_l2_order, :dpdx_error]])
println(convergence_table(refined)[:, [:dx, :profile_l2, :profile_l2_order, :v_l2,
                                      :v_l2_order, :dpdx_error]])
println(table[:, [:name, :n_particles_mean, :wall_time_stepping, :n_steps,
                  Symbol("flux_error_x0.5")]])
