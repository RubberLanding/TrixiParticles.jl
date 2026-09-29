mutable struct ParticleRefinementCallback{I, V, U}
    interval::I
    v_tmp::V
    u_tmp::U
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

    # Initialize the callback with `nothing` for the arrays initially
    refinement_callback = ParticleRefinementCallback(interval, nothing, nothing)

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
    t = integrator.t
    semi = integrator.p.semi
    v_ode, u_ode = integrator.u.x

    # Only allocate the backup arrays once, 
    # or if v_ode and u_ode expanded after a deep resize
    if isnothing(refinement_callback.v_tmp) || length(refinement_callback.v_tmp) < length(v_ode)
        refinement_callback.v_tmp = similar(v_ode)
        refinement_callback.u_tmp = similar(u_ode)
    end
    
    v_tmp = refinement_callback.v_tmp
    u_tmp = refinement_callback.u_tmp

    length_u_before = length(u_ode)

    refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, integrator, t)

    # Only resize the integrator if the refinement triggered a resize
    if length(u_ode) > length_u_before
        resize!(integrator, (length(v_ode), length(u_ode)))
    end

    # Force the solver to recalculate derivatives 
    # due to compaction, particle splitting and particle merging
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