using OrdinaryDiffEq
using OrdinaryDiffEqCore
using OrdinaryDiffEqLowStorageRK

function f!(du, u, p, t)
    du .= -0.1 .* u
end

u0 = [1.0, 2.0, 3.0]
tspan = (0.0, 1.0)
# p tracks: [has_refined, has_checked, refinement_step_number]
prob = ODEProblem(f!, u0, tspan, Any[false, false, 0])

# 1. Refinement Callback (Fires at t >= 0.5)
cond_refine(u, t, integrator) = t >= 0.5 && !integrator.p[1]
affect_refine!(integrator) = begin
    integrator.p[1] = true
    # Record the exact step number when refinement occurred
    integrator.p[3] = integrator.stats.naccept 
    println("\n--- Triggering Refinement at t = $(integrator.t) ---")
    
    new_size = 4
    old_u = copy(integrator.u)
    resize!(integrator.u, new_size)
    integrator.u[1] = old_u[1] 
    integrator.u[2] = old_u[2] 
    integrator.u[3] = old_u[2] 
    integrator.u[4] = old_u[3] 

    resize!(integrator, new_size)
    derivative_discontinuity!(integrator, true)
end
cb_refine = DiscreteCallback(cond_refine, affect_refine!, save_positions=(false, false))

# 2. Inspection Callback (Fires exactly ONE step after refinement)
# Strict integer check: only fire if the accepted step count has increased
cond_check(u, t, integrator) = integrator.p[1] && !integrator.p[2] && integrator.stats.naccept > integrator.p[3]
affect_check!(integrator) = begin
    integrator.p[2] = true
    println("--- State ONE step AFTER refinement (t = $(integrator.t)) ---")
    println("Current state (u):        ", integrator.u)
    println("Historical state (uprev): ", integrator.uprev)
    
    # Interpolation now spans a valid time step with properly initialized caches
    t_mid = integrator.t - (integrator.dt / 2)
    interpolated_u = integrator(t_mid)
    
    println("\n--- Interpolation Check at t = $t_mid ---")
    println("Interpolated state:       ", interpolated_u)
    println("--------------------------------------------------\n")
end
cb_check = DiscreteCallback(cond_check, affect_check!, save_positions=(false, false))

cb = CallbackSet(cb_refine, cb_check)
sol = solve(prob, RDPK3SpFSAL49(), callback=cb)