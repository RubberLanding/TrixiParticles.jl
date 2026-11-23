# E3: postprocessing of the single-cylinder vortex street.
#
#   julia --project=run experiments/e3_vortex_street/postprocess.jl [--full | --profile=gate]
#
# Integral quantities over the periodic window t ≥ t_start (the second half of the run):
#   cd_mean        mean drag coefficient
#   cl_amplitude   lift amplitude, sqrt(2) rms of the smoothed lift (see `shedding_quantities`)
#   cl_peak        (max - min) / 2 of the unsmoothed lift, includes the acoustic noise
#   st             Strouhal number f D / U, f from the zero crossings of the smoothed lift
# With fewer than 3 shedding periods in the window (e.g. the debug profile), `cl_amplitude`
# and `st` are `NaN` with a warning.
#
# Writes to `figure_data_directory("e3_vortex_street")`:
#   runs.csv                    all runs with cost and integral quantities
#   convergence_uniform.csv     uniform series, with the difference to the finest run
#   convergence_refined.csv     study A (the most common resolution ratio of the refined
#                               runs, vorticity criterion)
#   literature.csv              literature values at Re = 100, and
#   literature_range.csv        their range (band in E3b)
#   history_<run>.csv           Cd and Cl over time (E3a)
#   particles_vorticity_<kind>.png, particles_spacing_<kind>.png
#                               particle snapshots (point area ∝ mass), colored by
#                               vorticity and by spacing (E3c, E3d)
#   grid_vorticity_<kind>.png   interpolated vorticity (E3e), and the difference
#   grid_vorticity_difference.png
#   snapshot_limits.csv         runs, extents (in D, relative to the cylinder) and color
#                               limits (vorticity in U/D, spacing in D) of the images
# where <kind> is `uniform` and `refined`: the finest uniform run and the refined run
# (study A) with the same D/Δx_min.

include(joinpath(@__DIR__, "..", "common", "postprocess_util.jl"))

runs = load_runs(@__DIR__)
output = figure_data_directory("e3_vortex_street")

# ==========================================================================================
# ==== Integral quantities
# Centered moving average over `width` samples (shorter at the ends)
function moving_average(values, width)
    half = width ÷ 2
    return [sum(values[max(1, i - half):min(end, i + half)]) /
            length(max(1, i - half):min(length(values), i + half)) for i in eachindex(values)]
end

# The force signals carry acoustic noise of the weakly compressible model at frequencies
# far above the shedding frequency. The shedding is detected on the smoothed lift:
# - St from the upward zero crossings of the smoothed Cl - mean, with hysteresis
#   (a crossing only counts after the signal was below -0.25 rms), so that the noise
#   does not add crossings;
# - the Cl amplitude as sqrt(2) rms of the smoothed lift (exact for a sinusoid), and
#   `cl_peak` = (max - min) / 2 of the unsmoothed lift, which includes the noise.
function shedding_quantities(metrics, D, U; smoothing_time=0.05)
    t_start = metrics.time[end] / 2
    window = metrics[metrics.time .>= t_start, :]
    cl_mean = sum(window.cl) / nrow(window)

    dt = window.time[2] - window.time[1]
    cl = moving_average(window.cl .- cl_mean, max(1, round(Int, smoothing_time / dt)))
    rms = sqrt(sum(abs2, cl) / length(cl))

    # Upward zero crossings with hysteresis, linearly interpolated
    crossings = Float64[]
    armed = false
    for i in 1:(nrow(window) - 1)
        cl[i] < -0.25 * rms && (armed = true)
        if armed && cl[i] < 0 <= cl[i + 1]
            t0, t1 = window.time[i], window.time[i + 1]
            push!(crossings, t0 - cl[i] * (t1 - t0) / (cl[i + 1] - cl[i]))
            armed = false
        end
    end

    n_periods = max(length(crossings) - 1, 0)
    if n_periods < 3
        @warn "Only $n_periods shedding periods in t ≥ $t_start, no St and Cl amplitude"
        in_periods = trues(nrow(window))
        cl_amplitude = NaN
        st = NaN
    else
        # Means and amplitudes over whole periods only
        period = (crossings[end] - crossings[1]) / n_periods
        in_periods = crossings[1] .<= window.time .<= crossings[end]
        cl_amplitude = sqrt(2 * sum(abs2, cl[in_periods]) / count(in_periods))
        st = D / (period * U)

        # Without shedding, the crossings come from the noise
        if !(0.1 < st < 0.3)
            @warn "St = $st is far from the shedding frequency, the window is probably " *
                  "not periodic shedding"
        end
    end

    cd_mean = sum(window.cd[in_periods]) / count(in_periods)
    cd_pressure_mean = sum(window.cd_pressure[in_periods]) / count(in_periods)
    cl_peak = (maximum(window.cl[in_periods]) - minimum(window.cl[in_periods])) / 2

    return (; t_start, n_periods, cd_mean, cd_pressure_mean, cl_mean, cl_amplitude,
            cl_peak, st)
end

rows = map(runs) do run
    p = run.parameters
    q = shedding_quantities(run.metrics, p["diameter"], 1.0)

    (; name=run.name, refined=p["refined"], resolution=p["resolution"],
     min_spacing=p["min_spacing"], max_spacing=p["max_spacing"],
     dx_effective=effective_spacing(run),
     resolution_effective=p["diameter"] / effective_spacing(run),
     resolution_ratio=p["resolution_ratio"], criteria=p["criteria"],
     fluid_setup=p["fluid_setup"],
     n_particles_mean=sum(run.metrics.n_particles) / nrow(run.metrics),
     n_particles_max=maximum(run.metrics.n_particles),
     wall_time_total=run.summary["wall_time"],
     wall_time_stepping=run.metrics.wall_time[end],
     n_steps=run.summary["n_steps_accepted"], max_rss_mb=run.summary["max_rss_mb"],
     n_threads=run.summary["n_threads"], t_end=run.summary["t_end"],
     retcode=run.summary["retcode"], q...)
end

table = DataFrame(rows)
sort!(table, [:refined, :resolution, :resolution_ratio])
CSV.write(joinpath(output, "runs.csv"), table)

# ==========================================================================================
# ==== Convergence tables: self-convergence against the finest run of each series
function convergence_table(rows)
    rows = sort(rows, :resolution)
    # `resolution` is D / `min_spacing`, `resolution_effective` is D over the measured
    # smallest spacing (see `effective_spacing`)
    result = DataFrame(resolution=rows.resolution, dx=rows.min_spacing,
                       resolution_effective=rows.resolution_effective,
                       dx_effective=rows.dx_effective, fluid_setup=rows.fluid_setup)
    finest = nrow(rows)
    for column in (:cd_mean, :cl_amplitude, :st)
        result[!, column] = rows[!, column]
        result[!, Symbol(column, :_difference)] = abs.(rows[!, column] ./
                                                       rows[finest, column] .- 1)
    end

    return result
end

# Only the fluid setup that most runs use, so that the gate comparison of the setups
# does not mix into the convergence tables
main_setup = isempty(table) ? "" :
             argmax(setup -> count(==(setup), table.fluid_setup), unique(table.fluid_setup))
uniform = table[.!table.refined .& (table.fluid_setup .== main_setup), :]
refined_vorticity = table[table.refined .& (table.criteria .== "vorticity") .&
                          (table.fluid_setup .== main_setup), :]
ratio_a = isempty(refined_vorticity) ? 0 :
          argmax(r -> count(==(r), refined_vorticity.resolution_ratio),
                 unique(refined_vorticity.resolution_ratio))
refined_a = refined_vorticity[refined_vorticity.resolution_ratio .== ratio_a, :]

CSV.write(joinpath(output, "convergence_uniform.csv"), convergence_table(uniform))
CSV.write(joinpath(output, "convergence_refined.csv"), convergence_table(refined_a))

# Literature values for the unconfined cylinder at Re = 100 (2D simulations).
# TODO: check the values against the papers before using them in the thesis.
literature = DataFrame(source=["Park et al. 1998", "Liu et al. 1998",
                               "Braza et al. 1986"],
                       cd_mean=[1.33, 1.35, 1.364], cl_amplitude=[0.332, 0.339, NaN],
                       st=[0.165, 0.164, 0.160])
CSV.write(joinpath(output, "literature.csv"), literature)

# The range of the literature values (the band in E3b), one row with `<quantity>_low` and
# `<quantity>_high`
literature_range = DataFrame()
for column in (:cd_mean, :cl_amplitude, :st)
    values = filter(!isnan, literature[!, column])
    literature_range[!, Symbol(column, :_low)] = [minimum(values)]
    literature_range[!, Symbol(column, :_high)] = [maximum(values)]
end
CSV.write(joinpath(output, "literature_range.csv"), literature_range)

# ==========================================================================================
# ==== Force histories (E3a)
for run in runs
    history = select(run.metrics, :time, :cd, :cl, :cd_pressure, :cd_viscous,
                     :n_particles)
    CSV.write(joinpath(output, "history_$(run.name).csv"), history)
end

# ==========================================================================================
# ==== Snapshots (E3c, E3d, E3e): finest uniform run and the refined run of study A with
# the same D/Δx_min
common_resolutions = intersect(uniform.resolution, refined_a.resolution)

function last_snapshot(run, prefix)
    files = sort(filter(file -> startswith(file, prefix * "_t"), readdir(run.directory)))
    isempty(files) && return nothing

    times = [parse(Float64, file[(length(prefix) + 3):(end - 4)]) for file in files]

    return CSV.read(joinpath(run.directory, files[argmax(times)]), DataFrame)
end

# Vorticity on the regular grid by central differences (one-sided at the edges).
# The interpolation leaves out points without fluid neighbors (inside the cylinder),
# so missing points are NaN, and so is the vorticity next to them.
# Returns the vorticity at the points of `grid`.
function grid_vorticity(grid)
    xs = sort(unique(grid.x))
    ys = sort(unique(grid.y))
    index_x = Dict(x => i for (i, x) in enumerate(xs))
    index_y = Dict(y => j for (j, y) in enumerate(ys))

    vx = fill(NaN, length(xs), length(ys))
    vy = fill(NaN, length(xs), length(ys))
    for (x, y, u, v) in zip(grid.x, grid.y, grid.vx, grid.vy)
        vx[index_x[x], index_y[y]] = u
        vy[index_x[x], index_y[y]] = v
    end

    function derivative(field, coordinates, i, j, direction)
        k = direction == 1 ? i : j
        lo, hi = max(k - 1, 1), min(k + 1, length(coordinates))
        a, b = direction == 1 ? (field[lo, j], field[hi, j]) : (field[i, lo], field[i, hi])

        return (b - a) / (coordinates[hi] - coordinates[lo])
    end

    return [derivative(vy, xs, index_x[x], index_y[y], 1) -
            derivative(vx, ys, index_x[x], index_y[y], 2) for (x, y) in zip(grid.x, grid.y)]
end

"""
    write_particle_image(filename, particles, values; clims, colormap, window, pixels)

Particles in `window = (xmin, xmax, ymin, ymax)` as a PNG without axes, with the point
area proportional to the mass, scaled so that the particles of the largest mass just touch.
"""
function write_particle_image(filename, particles, values; clims, colormap, window,
                              pixels=1600)
    (xmin, xmax, ymin, ymax) = window
    inside = (xmin .<= particles.x .<= xmax) .& (ymin .<= particles.y .<= ymax)
    p = particles[inside, :]
    v = values[inside]

    width = pixels
    height = round(Int, pixels * (ymax - ymin) / (xmax - xmin))
    # Spacing sqrt(m / ρ_0) of each particle in pixels, so that the point area is
    # proportional to the mass. The marker size of Plots is a radius.
    marker = sqrt.(p.mass ./ 1000.0) ./ (xmax - xmin) .* width

    plot_ = scatter(p.x, p.y; zcolor=v, clims, color=colormap, markersize=marker ./ 2,
                    markerstrokewidth=0, legend=false, colorbar=false, axis=false,
                    ticks=false, grid=false, framestyle=:none, xlims=(xmin, xmax),
                    ylims=(ymin, ymax), size=(width, height), margin=0Plots.mm,
                    background_color=:transparent)
    savefig(plot_, filename)

    return filename
end

if isempty(common_resolutions)
    @warn "No uniform and refined run with the same D/Δx_min, no snapshots"
    limits = DataFrame()
else
    resolution = maximum(common_resolutions)
    kinds = ("uniform", "refined")
    snapshot_runs = Dict("uniform" => uniform[uniform.resolution .== resolution, :name][1],
                         "refined" => refined_a[refined_a.resolution .== resolution,
                                                :name][1])
    run_by_name = Dict(run.name => run for run in runs)

    particles = Dict(kind => last_snapshot(run_by_name[snapshot_runs[kind]], "particles")
                     for kind in kinds)
    grids = Dict(kind => last_snapshot(run_by_name[snapshot_runs[kind]], "grid")
                 for kind in kinds)

    # The window of the particle images: the snapshot grid
    grid = grids["uniform"]
    window = (minimum(grid.x), maximum(grid.x), minimum(grid.y), maximum(grid.y))

    # Shared, symmetric vorticity limits: the 99th percentile of |ω| of the uniform run,
    # so that single noisy particles do not set the scale
    function percentile(values, q)
        sorted = sort(filter(!isnan, values))
        return sorted[clamp(round(Int, q * length(sorted)), 1, length(sorted))]
    end
    omega_max = percentile(abs.(particles["uniform"].vorticity), 0.99)
    spacing(table) = sqrt.(table.mass ./ 1000.0)
    spacing_limits = (minimum(minimum(spacing(particles[kind])) for kind in kinds),
                      maximum(maximum(spacing(particles[kind])) for kind in kinds))

    grid_vorticities = Dict(kind => grid_vorticity(grids[kind]) for kind in kinds)

    for kind in kinds
        write_particle_image(joinpath(output, "particles_vorticity_$kind.png"),
                             particles[kind], particles[kind].vorticity;
                             clims=(-omega_max, omega_max), colormap=:RdBu, window)
        write_particle_image(joinpath(output, "particles_spacing_$kind.png"),
                             particles[kind], spacing(particles[kind]);
                             clims=spacing_limits, colormap=:viridis, window)
        g = grids[kind]
        write_image(joinpath(output, "grid_vorticity_$kind.png"), g.x, g.y,
                    grid_vorticities[kind]; clims=(-omega_max, omega_max), colormap=:RdBu)
    end

    # The grids can miss different points next to the cylinder, so match the points by
    # their coordinates (the grids have the same points otherwise)
    point_key(x, y) = (round(x, digits=10), round(y, digits=10))
    refined_vorticity = Dict(point_key(x, y) => value
                             for (x, y, value) in zip(grids["refined"].x,
                                                      grids["refined"].y,
                                                      grid_vorticities["refined"]))
    difference = [get(refined_vorticity, point_key(x, y), NaN) - value
                  for (x, y, value) in zip(grid.x, grid.y, grid_vorticities["uniform"])]
    difference_max = percentile(abs.(difference), 0.99)
    write_image(joinpath(output, "grid_vorticity_difference.png"), grid.x, grid.y,
                difference; clims=(-difference_max, difference_max), colormap=:RdBu)

    # Limits for pgfplots in dimensionless units: lengths in D relative to the cylinder
    # center, vorticity in U/D, spacing in D
    run = run_by_name[snapshot_runs["uniform"]]
    D = run.parameters["diameter"]
    center = run.parameters["cylinder_center"]
    limits = DataFrame(uniform=snapshot_runs["uniform"], refined=snapshot_runs["refined"],
                       resolution=resolution,
                       xmin=(window[1] - center[1]) / D, xmax=(window[2] - center[1]) / D,
                       ymin=(window[3] - center[2]) / D, ymax=(window[4] - center[2]) / D,
                       vorticity_max=omega_max * D, difference_max=difference_max * D,
                       spacing_min=spacing_limits[1] / D,
                       spacing_max=spacing_limits[2] / D)
end

CSV.write(joinpath(output, "snapshot_limits.csv"), limits)

println("E3 postprocessing ($(experiment_profile())) written to ", abspath(output))
println(table[:, [:name, :n_particles_mean, :wall_time_stepping, :t_end, :n_periods,
                  :cd_mean, :cl_amplitude, :st]])
