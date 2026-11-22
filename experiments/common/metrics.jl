# Metrics of a run: time series for the `PostprocessCallback` and a summary at the end.

using TrixiParticles
using JSON

const TP = TrixiParticles

"""
    run_metrics(; fluid_index=1)

Custom quantities for the `PostprocessCallback`, written for the fluid system only:
number of active particles, total mass, total momentum, smallest and largest smoothing
length, wall time and peak memory. The fluid system is identified by its index in the
semidiscretization, so that the callback can be created before the semidiscretization.

```julia
PostprocessCallback(; dt=0.01, run_metrics()...)
```

`wall_time` is measured from the first evaluation (the initial state), so it excludes
the setup, but includes the compilation of the first time steps unless the run was
preceded by a warm-up run in the same Julia session.
"""
function run_metrics(; fluid_index=1)
    start_time = Ref{UInt64}(0)

    # Wrap a function of the fluid system, so that the other systems return `nothing`
    function fluid_only(f)
        return function (system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            TP.system_indices(system, semi) == fluid_index || return nothing

            return f(system, v_ode, semi)
        end
    end

    n_particles = fluid_only() do system, v_ode, semi
        return length(TP.each_active_particle(system))
    end

    total_mass = fluid_only() do system, v_ode, semi
        return sum(particle -> TP.hydrodynamic_mass(system, particle),
                   TP.each_active_particle(system))
    end

    function momentum(dim)
        return fluid_only() do system, v_ode, semi
            v = TP.wrap_v(v_ode, system, semi)
            return sum(particle -> TP.hydrodynamic_mass(system, particle) *
                                   TP.current_velocity(v, system, particle)[dim],
                       TP.each_active_particle(system))
        end
    end

    h_min = fluid_only() do system, v_ode, semi
        return minimum(particle -> TP.smoothing_length(system, particle),
                       TP.each_active_particle(system))
    end

    h_max = fluid_only() do system, v_ode, semi
        return maximum(particle -> TP.smoothing_length(system, particle),
                       TP.each_active_particle(system))
    end

    wall_time = fluid_only() do system, v_ode, semi
        start_time[] == 0 && (start_time[] = time_ns())
        return (time_ns() - start_time[]) / 1e9
    end

    max_rss_mb = fluid_only() do system, v_ode, semi
        return Sys.maxrss() / 2^20
    end

    return (; n_particles, total_mass, momentum_x=momentum(1), momentum_y=momentum(2),
            h_min, h_max, wall_time, max_rss_mb)
end

"""
    write_run_summary(output_directory, sol, semi, fluid_system; wall_time, parameters)

Write `summary.json` (steps, wall time, peak memory, final particle count and the
parameters of the run) and `timer.json`/`timer.txt` (the TrixiParticles timer).
`wall_time` is the time of `solve` in seconds, e.g. from `@elapsed`.
"""
function write_run_summary(output_directory, sol, semi, fluid_system; wall_time,
                           parameters=Dict())
    mkpath(output_directory)

    n_final = length(TP.each_active_particle(fluid_system))

    summary = Dict("parameters" => parameters,
                   "wall_time" => wall_time,
                   "n_steps_accepted" => sol.stats.naccept,
                   "n_steps_rejected" => sol.stats.nreject,
                   "t_end" => sol.t[end],
                   "retcode" => string(sol.retcode),
                   "n_particles_final" => n_final,
                   "max_rss_mb" => Sys.maxrss() / 2^20,
                   "n_threads" => Threads.nthreads())

    open(joinpath(output_directory, "summary.json"), "w") do io
        JSON.print(io, summary, 2)
    end

    open(joinpath(output_directory, "timer.json"), "w") do io
        JSON.print(io, TP.TimerOutputs.todict(TP.timer()), 2)
    end

    open(joinpath(output_directory, "timer.txt"), "w") do io
        TP.print_timer(io, TP.timer())
    end

    return summary
end
