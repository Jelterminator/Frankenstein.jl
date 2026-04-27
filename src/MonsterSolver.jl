module MonsterSolver

using SciMLBase
using ADTypes
using SparseArrays
using SparseDiffTools
using SparseMatrixColorings
using ..FCore
using OrdinaryDiffEq
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
function monster_solve!(prob::SciMLBase.ODEProblem, Fs::FCore.FrankensteinSolver; kwargs...)
    # 0. Separate Frankenstein keywords from SciML keywords
    frankenstein_kwargs = filter(x -> x.first in (:prefer_memory, :prefer_stability, :max_retries, :ad_available), kwargs)
    sciml_kwargs = filter(x -> !(x.first in (:prefer_memory, :prefer_stability, :max_retries, :ad_available)), kwargs)

    # 1. Initial Analysis
    analysis = Analysis.analyze_system_structure(prob)
    
    # 2. Robust preparation: Standardize the ODEFunction
    raw_f = prob.f isa SciMLBase.ODEFunction ? prob.f.f : prob.f
    Fs.original_f = raw_f
    
    # Detect the correct pattern to use
    new_pattern = if analysis.is_sparse && analysis.sparsity_pattern !== nothing
        p = analysis.sparsity_pattern
        p isa SparseArrays.SparseMatrixCSC ? p : sparse(p)
    else
        hasproperty(prob.f, :jac_prototype) ? prob.f.jac_prototype : nothing
    end

    # Use Non-Destructive remake [User Fix 4]
    new_f = if prob.f isa SciMLBase.ODEFunction
        SciMLBase.remake(prob.f; 
                         f = raw_f,
                         jac_prototype = new_pattern,
                         sparsity = new_pattern !== nothing ? new_pattern : (hasproperty(prob.f, :sparsity) ? prob.f.sparsity : nothing))
    else
        SciMLBase.ODEFunction(raw_f; 
                              jac_prototype = new_pattern,
                              sparsity = new_pattern)
    end

    prob = SciMLBase.remake(prob, f = new_f)
    
    # 3. Initial Configuration
    rec = select_algorithm(analysis; frankenstein_kwargs...)
    analysis.current_category = rec.category
    
    # Select technical backends
    ad_available = get(frankenstein_kwargs, :ad_available, 
        [ADTypes.AutoForwardDiff(), 
         ADTypes.AutoEnzyme(), 
         ADTypes.AutoSparse(ADTypes.AutoForwardDiff()), 
         ADTypes.AutoSymbolics(), # FIX: Typo
         ADTypes.AutoFiniteDiff()])
    
    backend_selection = Backends.choose_backend(analysis, ad_available; 
                                             disabled_backends=Fs.disabled_backends)
    
    cfg = create_solver_configuration(rec, analysis, backend_selection)
    
    # Initialize Adaptation Controller
    controller = AdaptationController()
    register_strategy!(controller, PerformanceAdaptationStrategy(), 1.0)
    register_strategy!(controller, StabilityAdaptationStrategy(), 1.5)
    
    @info "[Frankenstein] Initializing with $(typeof(cfg.algorithm))"
    @info "[Frankenstein] Backend selection: $(backend_selection.selection_rationale)"
    
    # 4. Solver initialization
    integrator = try
        init(prob, cfg.algorithm; reltol=cfg.reltol, abstol=cfg.abstol, sciml_kwargs...)
    catch err
        @warn "[Frankenstein] Initial solver configuration failed: $err. Pivoting..."
        
        current_backend_name = string(typeof(backend_selection.ad_backend))
        Fs.disabled_backends[current_backend_name] = 999999
        
        ad_available_filtered = filter(b -> string(typeof(b)) != current_backend_name, ad_available)
        backend_selection = Backends.choose_backend(analysis, ad_available_filtered; 
            disabled_backends=Fs.disabled_backends)
        
        cfg = create_solver_configuration(rec, analysis, backend_selection)
        init(prob, cfg.algorithm; reltol=cfg.reltol, abstol=cfg.abstol, sciml_kwargs...)
    end

    # 5. Main integration loop
    max_retries = get(frankenstein_kwargs, :max_retries, 5)
    Fs.recovery_attempts = 0 # FIX: Initialization [User Fix 3]
    
    while integrator.t < prob.tspan[end]
        try
            step!(integrator)
            
            # Pulse check & Adaptive surgery
            # Construct StepInfo for the brain
            step_info = StepInfo(integrator.u, integrator.uprev, integrator.dt, integrator.dtpropose, 
                                 integrator.stats.nreject, integrator.stats.naccept + integrator.stats.nreject, integrator.t, 
                                 integrator.p, integrator.sol.prob)
            
            if Analysis.light_pulse(analysis, step_info)
                @info "[Frankenstein] Pulse detected anomaly at t=$(integrator.t). Performing heavy diagnostics..."
                
                # Perform deep diagnostic
                Analysis.heavy_diagnostic!(analysis, step_info)
                
                # New recommendation based on current state
                new_rec = Adaptation.adapt!(controller, analysis, step_info)
                
                # COOL-DOWN: Don't trigger surgery if we just had one (< 50 steps ago)
                # unless there is a rejection (emergency)
                cooldown_active = (analysis.current_step - analysis.last_surgery_step) < 50
                
                if new_rec !== nothing && !(cfg.algorithm isa new_rec.algorithm)
                    if cooldown_active && step_info.rejects <= analysis.last_reject_count
                        @debug "[Frankenstein] Surgery recommended but blocked by cooldown ($(analysis.current_step - analysis.last_surgery_step) steps since last)."
                    else
                        @info "✂️ [Frankenstein] SURGERY RECOMMENDED: $(typeof(cfg.algorithm)) ➔ $(new_rec.algorithm)"
                        @info "🏥 [Frankenstein] Rationale: $(new_rec.description)"
                        
                        new_backend_selection = Backends.choose_backend(analysis, ad_available; 
                                                                    disabled_backends=Fs.disabled_backends)
                        
                        new_cfg = create_solver_configuration(new_rec, analysis, new_backend_selection)
                        
                        # Perform the surgical swap
                        integrator = reinit_with_new_alg(integrator, Fs, new_cfg.algorithm, integrator.u, integrator.t, sciml_kwargs)
                        cfg = new_cfg # Update current config
                        analysis.current_category = new_rec.category
                        @info "💉 [Frankenstein] Surgery Successful! Patient stabilized with: $(typeof(cfg.algorithm))"
                        
                        # Reset cooldown and track surgery
                        analysis.last_surgery_step = analysis.current_step
                        analysis.diagnostic_cooldown = 50 
                        analysis.watchdog_interval = 10000 # Reset watchdog on surgery
                    end
                else
                    # BACK-OFF: If we performed a diagnostic but decided NO surgery is needed,
                    # increase the cooldown and watchdog interval so we don't spam diagnostics for a stable system.
                    analysis.diagnostic_cooldown = min(1000, analysis.diagnostic_cooldown * 2)
                    analysis.watchdog_interval = min(100000, analysis.watchdog_interval * 2) # Iteratively longer watchdog
                    @debug "[Frankenstein] No surgery needed. Backing off: cooldown=$(analysis.diagnostic_cooldown), watchdog=$(analysis.watchdog_interval)"
                end
            end
            
        catch err
            @error "[Frankenstein] Step failed with error: $err"
            
            if Fs.recovery_attempts >= max_retries
                @error "[Frankenstein] Max retries reached. Simulation aborted."
                throw(err)
            end
            
            Fs.recovery_attempts += 1
            @warn "[Frankenstein] Recovery attempt $(Fs.recovery_attempts)/$max_retries..."
            
            if occursin("DimensionMismatch", string(err)) || occursin("sparsity", string(err))
                current_backend_name = string(typeof(backend_selection.ad_backend))
                Fs.disabled_backends[current_backend_name] = 999999
                @warn "[Frankenstein] Sparse AD mismatch detected. Disabling backend and performing Surgery..."
            end
            
            # Emergency diagnostic and pivot
            step_info = StepInfo(integrator.u, integrator.uprev, integrator.dt, integrator.dtpropose, 
                                 integrator.stats.nreject, integrator.stats.naccept + integrator.stats.nreject, integrator.t, 
                                 integrator.p, integrator.sol.prob)
                                 
            Analysis.heavy_diagnostic!(analysis, step_info)
            new_rec = Adaptation.adapt!(controller, analysis, step_info)
            
            # Re-select and retry
            ad_available_filtered = filter(b -> !(string(typeof(b)) in keys(Fs.disabled_backends)), ad_available)
            backend_selection = Backends.choose_backend(analysis, ad_available_filtered; 
                disabled_backends=Fs.disabled_backends)
            
            # Use new_rec if adaptation recommended something, otherwise fallback to previous rec
            actual_rec = new_rec !== nothing ? new_rec : rec
            cfg = create_solver_configuration(actual_rec, analysis, backend_selection)
            analysis.current_category = actual_rec.category
            
            integrator = reinit_with_new_alg(integrator, Fs, cfg.algorithm, integrator.u, integrator.t, sciml_kwargs)
            @info "[Frankenstein] Attempting recovery with: $(backend_selection.selection_rationale)"
        end
    end

    return SciMLBase.solution_new_retcode(integrator.sol, :Success)
end

# Internal helpers for surgery
function reinit_with_new_alg(integrator, Fs::FrankensteinSolver, new_alg, u, t, sciml_kwargs)
    prob = integrator.sol.prob
    
    # Use the original stored function to avoid FunctionWrappersWrappers conflicts
    raw_f = Fs.original_f
    
    clean_f = SciMLBase.ODEFunction{SciMLBase.isinplace(prob.f)}(raw_f; 
                                   jac_prototype = hasproperty(prob.f, :jac_prototype) ? prob.f.jac_prototype : nothing,
                                   sparsity = hasproperty(prob.f, :sparsity) ? prob.f.sparsity : nothing)
    
    new_prob = SciMLBase.remake(prob, f = clean_f, u0 = u, tspan = (t, prob.tspan[end]))
    
    # Preserve key properties
    reltol = integrator.opts.reltol
    abstol = integrator.opts.abstol
    
    @info "💉 [Frankenstein] Re-initializing with $(SciMLBase.isinplace(new_prob.f) ? "in-place" : "out-of-place") function and $(typeof(new_alg))"
    return init(new_prob, new_alg; reltol=reltol, abstol=abstol, sciml_kwargs...)
end

function should_trigger_adaptation(controller, integrator)
    # Placeholder for pulse logic
    # e.g. monitor step size drops or reject rates
    return false
end

function diagnose_system(integrator)
    # Placeholder for heavy diagnostics
    return nothing
end

function recommend_adaptation(controller, integrator, diagnosis)
    # Placeholder for adaptation logic
    return nothing
end

end # module

# Dispatch is handled in Frankenstein.jl
