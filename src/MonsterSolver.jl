module MonsterSolver

using SciMLBase
using ADTypes
using SparseDiffTools
using SparseMatrixColorings
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
    
    # Robust preparation: Standardize the ODEFunction and preserve existing jac_prototype.
    # We also synchronize the sparsity pattern if detected.
    raw_f = prob.f isa SciMLBase.ODEFunction ? prob.f.f : prob.f
    
    # Safely get existing jac_prototype
    jp_existing = hasproperty(prob.f, :jac_prototype) ? prob.f.jac_prototype : nothing
    
    new_f = SciMLBase.ODEFunction(raw_f; 
                       jac_prototype = (analysis.is_sparse && analysis.sparsity_pattern !== nothing) ? 
                                       analysis.sparsity_pattern : jp_existing)
    prob = SciMLBase.remake(prob, f = new_f)
    
    if analysis.is_sparse && analysis.sparsity_pattern !== nothing
        @info "[Frankenstein] Synchronized detected sparsity pattern and disabled FunctionWrappers."
    else
        @info "[Frankenstein] Prepared ODEFunction for adaptive backends (FunctionWrappers disabled)."
    end

    rec = select_algorithm(analysis; frankenstein_kwargs...)
    
    # Select technical backends (AD and Linear Solver)
    ad_available = get(frankenstein_kwargs, :ad_available, 
        [AutoForwardDiff(), 
         AutoEnzyme(), 
         AutoSparseForwardDiff(), 
         AutoSymbolics(), 
         AutoFiniteDiff()])
    backend_selection = Backends.choose_backend(analysis, ad_available; is_external_solver=rec.is_sundials)
    
    cfg = create_solver_configuration(rec, analysis, backend_selection)
    
    # Initialize Adaptation Controller
    controller = AdaptationController()
    register_strategy!(controller, PerformanceAdaptationStrategy(), 1.0)
    register_strategy!(controller, StabilityAdaptationStrategy(), 1.5)
    
    @info "[Frankenstein] Initializing with $(typeof(cfg.algorithm))"
    @info "[Frankenstein] Backend selection: $(backend_selection.selection_rationale)"
    
    # 2. Initialize the first integrator
    # 2. Solver initialization with error handling
    integrator = try
        init(prob, cfg.algorithm; 
             reltol=cfg.reltol, abstol=cfg.abstol, sciml_kwargs...)
    catch err
        @warn "[Frankenstein] Initial solver configuration failed: $err"
        if occursin("FunctionWrapper", string(err)) || occursin("No matching function wrapper", string(err)) || occursin("Differentiation", string(err))
            @warn "[Frankenstein] $(typeof(backend_selection.ad_backend)) failure during init. Retrying with next best backend..."
            
            # Record the failure and re-select
            current_backend_name = string(typeof(backend_selection.ad_backend))
            Fs.disabled_backends[current_backend_name] = 999999 # Disable practically forever for this run
            
            ad_available_filtered = filter(b -> string(typeof(b)) != current_backend_name, ad_available)
            backend_selection = Backends.choose_backend(analysis, ad_available_filtered; 
                is_external_solver=rec.is_sundials,
                disabled_backends=Fs.disabled_backends)
            cfg = create_solver_configuration(rec, analysis, backend_selection)
            
            @info "[Frankenstein] Recovered with $(typeof(backend_selection.ad_backend))"
            init(prob, cfg.algorithm; reltol=cfg.reltol, abstol=cfg.abstol, sciml_kwargs...)
        else
            rethrow(err)
        end
    end
    max_retries = get(frankenstein_kwargs, :max_retries, 3)
    
    # 3. Main stepping loop
    while integrator.t < integrator.sol.prob.tspan[2]
        # Perform one or more steps
        try
            step!(integrator)
        catch err
            @error "[Frankenstein] Step failed with error: $err"
            
            # Check if this was a differentiation error (e.g. Enzyme crash)
            if occursin("Differentiation", string(typeof(err))) || occursin("Autodiff", string(typeof(err)))
                @warn "[Frankenstein] Differentiation failure detected. Disabling backend for 1000 steps and performing Surgery..."
                
                # Record the disablement (current step + 1000)
                current_backend_name = string(typeof(backend_selection.ad_backend))
                Fs.disabled_backends[current_backend_name] = analysis.current_step + 1000
                
                # Re-select backend
                backend_selection = Backends.choose_backend(analysis, ad_available; 
                    is_external_solver=rec.is_sundials,
                    disabled_backends=Fs.disabled_backends)
                cfg = create_solver_configuration(rec, analysis, backend_selection)
                
                @info "[Frankenstein] Surgery Successful! Pivoting to: $(typeof(cfg.algorithm))"
                @info "[Frankenstein] New Backend: $(backend_selection.selection_rationale)"
                
                integrator = reinit_with_new_alg(integrator, cfg.algorithm, integrator.u, integrator.t)
                continue
            end

            if retry_count < max_retries
                integrator = handle_instability!(integrator, retry_count)
                retry_count += 1
                continue
            else
                rethrow(err)
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
        stats = hasproperty(integrator, :stats) ? integrator.stats : (hasproperty(integrator, :destats) ? integrator.destats : nothing)
        rejects = stats !== nothing && hasproperty(stats, :nreject) ? stats.nreject : 0
        nsteps = stats !== nothing && hasproperty(stats, :naccept) ? stats.naccept : (stats !== nothing && hasproperty(stats, :nsteps) ? stats.nsteps : 0)
        
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
            nsteps,
            integrator.t,
            integrator.p,
            integrator.sol.prob
        )
        
        # TIERED DIAGNOSTICS:
        # Step 1: Light Pulse (Every Step)
        if light_pulse(analysis, step_info)
            # Step 2: Heavy Diagnosis (Triggered)
            heavy_diagnostic!(analysis, step_info)
            
            # Step 3: Adaptation & Potential Surgery
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
