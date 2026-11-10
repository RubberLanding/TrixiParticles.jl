struct ParticleRefinementCallback{I}
    interval::I
end

function ParticleRefinementCallback(; interval::Integer=-1, dt=0.0)
    if dt > 0 && interval !== -1
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt > 0
        interval = Float64(dt)
    elseif interval == -1
        interval = 1
    end

    refinement_callback = ParticleRefinementCallback(interval)

    if dt > 0
        return PeriodicCallback(refinement_callback, dt,
                                save_positions=(false, false))
    else
        return DiscreteCallback(refinement_callback, refinement_callback,
                                save_positions=(false, false))
    end
end

# initialize
function initial_refinement!(cb, u, t, integrator)
    initial_refinement!(cb.affect!, u, t, integrator)
end

function initial_refinement!(cb::ParticleRefinementCallback, u, t, integrator)
    cb(integrator)
end

# condition
function (refinement_callback::ParticleRefinementCallback)(u, t, integrator)
    (; interval) = refinement_callback
    return condition_integrator_interval(integrator, interval)
end

# affect
function (refinement_callback::ParticleRefinementCallback)(integrator)
    semi = integrator.p.semi
    v_ode, u_ode = integrator.u.x

    length_u_before = length(u_ode)

    refinement!(semi, v_ode, u_ode)

    # Only resize the integrator if the refinement resized the systems
    if length(u_ode) > length_u_before
        resize!(integrator, (length(v_ode), length(u_ode)))
    end

    # Force the solver to recalculate derivatives
    # due to particle splitting, merging and shifting
    SciMLBase.derivative_discontinuity!(integrator, true)

    return integrator
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb
    print(io, "ParticleRefinementCallback(interval=", (cb.affect!.interval), ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:ParticleRefinementCallback}})
    @nospecialize cb
    print(io, "ParticleRefinementCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb

    if get(io, :compact, false)
        show(io, cb)
    else
        refinement_cb = cb.affect!
        setup = [
            "interval" => refinement_cb.interval
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:ParticleRefinementCallback}})
    @nospecialize cb

    if get(io, :compact, false)
        show(io, cb)
    else
        refinement_cb = cb.affect!.affect!
        setup = [
            "dt" => refinement_cb.interval
        ]
        summary_box(io, "ParticleRefinementCallback", setup)
    end
end
