module MonsterSolver

using SciMLBase
using ADTypes
using ..FCore
using ..Analysis
using ..Solvers
using ..Adaptation
using ..Backends

export monster_solve!

"""
    monster_solve!(prob::ODEProblem, Fs::FrankensteinSolver; kwargs...)

The central adaptive loop for Frankenstein. It initializes an integrator,
steps through the problem, and performs 'surgery' (algorithm swapping)
when the adaptation logic dictates.
"""
function monster_solve!(prob::ODEProblem, Fs::FrankensteinSolver; kwargs...)
    # 0. Separate Frankenstein keywords from SciML keywords
    frankenstein_kwargs = filter(x -> x.first in (:prefer_memory, :prefer_stability, :max_retries, :ad_available), kwargs)
    sciml_kwargs = filter(x -> !(x.first in (:prefer_memory, :prefer_stability, :max_retries, :ad_available)), kwargs)

    # 1. Initial analysis and selection
    analysis = analyze_system_structure(prob)
    rec = select_algorithm(analysis; frankenstein_kwargs...)
    
    # Select technical backends (AD and Linear Solver)
    ad_available = get(frankenstein_kwargs, :ad_available, 
        [AutoForwardDiff(), AutoEnzyme(), AutoSparseForwardDiff(), AutoSymbolic(), AutoFiniteDiff()])
    backend_selection = Backends.choose_backend(analysis, ad_available)
    
    cfg = create_solver_configuration(rec, analysis, backend_selection)
    
    # Initialize Adaptation Controller
    controller = AdaptationController()
    register_strategy!(controller, PerformanceAdaptationStrategy(), 1.0)
    register_strategy!(controller, StabilityAdaptationStrategy(), 1.5)
    
    @info "[Frankenstein] Initializing with $(typeof(cfg.algorithm))"
    @info "[Frankenstein] Backend selection: $(backend_selection.selection_rationale)"
    
    # 2. Initialize the first integrator
    integrator = init(prob, cfg.algorithm; 
        reltol=cfg.reltol, abstol=cfg.abstol, sciml_kwargs...)

    
    retry_count = 0
    max_retries = get(frankenstein_kwargs, :max_retries, 3)
    
    # 3. Main stepping loop
    while integrator.t < integrator.sol.prob.tspan[2]
        # Perform one or more steps
        try
            step!(integrator)
        catch e
            @error "[Frankenstein] Step failed with error: $e"
            if retry_count < max_retries
                integrator = handle_instability!(integrator, retry_count)
                retry_count += 1
                continue
            else
                rethrow(e)
            end
        end
        
        # Check for solver-indicated failures or excessive rejections
        if integrator.sol.retcode != ReturnCode.Default && integrator.sol.retcode != ReturnCode.Success
            if retry_count < max_retries
                @warn "[Frankenstein] Solver reported $(integrator.sol.retcode). Retrying..."
                integrator = handle_instability!(integrator, retry_count)
                retry_count += 1
                continue
            else
                break
            end
        end

        # 4. Periodic Analysis & Adaptation Update
        rejects = hasproperty(integrator, :destats) ? integrator.destats.nreject : 0
        
        du_buffer = similar(integrator.u)
        integrator.f(du_buffer, integrator.u, integrator.p, integrator.t)
        
        # Safe access for dt and dtcache (Sundials etc might not have them)
        curr_dt = hasproperty(integrator, :dt) ? integrator.dt : (hasproperty(integrator, :dt_prev) ? integrator.dt_prev : 0.0)
        curr_dtcache = hasproperty(integrator, :dtcache) ? integrator.dtcache : curr_dt

        step_info = FCore.StepInfo(
            integrator.u, 
            du_buffer, 
            curr_dt, 
            curr_dtcache,
            rejects,
            integrator.t,
            integrator.p,
            integrator.sol.prob
        )
        
        updates = needs_analysis_update!(analysis, step_info)
        
        if any(updates)
            # Perform deeper updates
            if updates.stiffness
                Analysis.Stiffness.update_stiffness!(analysis, step_info)
            end
            
            # 5. Check for adaptation via Controller
            new_rec = adapt!(controller, analysis, step_info)
            
            if new_rec !== nothing && typeof(new_rec.algorithm) != typeof(cfg.algorithm)
                @info "[Frankenstein] Surgery required! Swapping $(typeof(cfg.algorithm)) -> $(typeof(new_rec.algorithm)) at t=$(integrator.t)"
                
                # Store state
                u_fixed = copy(integrator.u)
                t_fixed = integrator.t
                
                # Reinitialize current integrator with new algorithm
                backend_selection = Backends.choose_backend(analysis, ad_available)
                new_cfg = create_solver_configuration(new_rec, analysis, backend_selection)
                cfg = new_cfg
                rec = new_rec
                
                integrator = reinit_with_new_alg(integrator, cfg.algorithm, u_fixed, t_fixed)

                # Reset retry count on successful adaptation
                retry_count = 0
            end
        end
        
        # Monitor for high rejection rate
        if rejects > 5 && retry_count < max_retries
             @warn "[Frankenstein] High rejection rate detected ($rejects). Tuning tolerances..."
             integrator = handle_instability!(integrator, retry_count)
             retry_count += 1
        end
    end
    
    return solve!(integrator)
end

"""
    SciMLBase.solve(prob::ODEProblem, Fs::FrankensteinSolver; kwargs...)

Extends SciMLBase.solve to dispatch to the Monster Solver logic.
"""
function SciMLBase.solve(prob::ODEProblem, Fs::FrankensteinSolver; kwargs...)
    return monster_solve!(prob, Fs; kwargs...)
end

function handle_instability!(integrator, retry_count)
    u = copy(integrator.u)
    t = integrator.t
    prob = integrator.sol.prob
    
    # Tighten tolerances
    new_reltol = integrator.opts.reltol * 0.1
    new_abstol = integrator.opts.abstol * 0.1
    
    @info "[Frankenstein] Retrying instability at t=$t. New reltol=$new_reltol"
    
    new_alg = integrator.alg
    
    # Reinitialize with tighter tolerances and more iterations
    new_prob = ODEProblem(prob.f, u, (t, prob.tspan[2]), prob.p)
    return init(new_prob, new_alg; 
        reltol=new_reltol, 
        abstol=new_abstol,
        maxiters=Int(1e7),
        callback=integrator.opts.callback)
end

function reinit_with_new_alg(integrator, new_alg, u, t)
    prob = integrator.sol.prob
    new_prob = ODEProblem(prob.f, u, (t, prob.tspan[2]), prob.p)
    return init(new_prob, new_alg; 
        reltol=integrator.opts.reltol, 
        abstol=integrator.opts.abstol,
        callback=integrator.opts.callback)
end

end # module
