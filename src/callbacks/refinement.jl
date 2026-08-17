struct ParticleRefinementCallback{I}
    interval::I
end

function ParticleRefinementCallback(; interval::Integer=-1, dt=0.0)
    if dt > 0 && interval !== -1
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    # Update in intervals in terms of simulation time
    if dt > 0
        interval = Float64(dt)

        # Update every time step (default)
    elseif interval == -1
        interval = 1
    end

    refinement_callback = ParticleRefinementCallback(interval)

    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(refinement_callback, dt,
                                save_positions=(false, false))
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(refinement_callback, refinement_callback,
                                save_positions=(false, false))
    end
end

# initialize
function initial_refinement!(cb, u, t, integrator)
    # The `ParticleRefinementCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
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
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs!(semi, u_ode)

    v_tmp = similar(v_ode)
    u_tmp = similar(u_ode)

    v_tmp .= v_ode
    u_tmp .= u_ode

    # TODO
    refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, integrator, t)

    resize!(integrator, (length(v_ode), length(u_ode)))
    
    # TODO: Check if this is needed after the resize of the integrator
    # integrator.uprev .= integrator.u
    # OrdinaryDiffEqCore.auto_dt_reset!(integrator)

    # Tell OrdinaryDiffEq that u has been modified
    SciMLBase.u_modified!(integrator, true)

    # With SciMLBase v3, `u_modified!` is technically deprecated.  
    # SciMLBase.derivative_discontinuity!(integrator, true)
    
    return integrator
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleRefinementCallback(interval=", (cb.affect!.interval), ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:ParticleRefinementCallback}})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleRefinementCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:ParticleRefinementCallback})
    @nospecialize cb # reduce precompilation time

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
    @nospecialize cb # reduce precompilation time

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