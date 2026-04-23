module CompositeSolvers

using DifferentialEquations
using OrdinaryDiffEq
using Sundials
using LinearSolve
using SparseArrays
using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, COMPOSITE

export CompositeSolverStrategy, get_composite_recommendations, is_mixed_stiffness_problem,
       estimate_explicit_fraction, analyze_splitting_potential, configure_composite_solver,
       recommend_composite_solver, suggest_problem_splitting

# Implementation
#==============================================================================#
# Composite Solver Strategy Implementation
#==============================================================================#

struct CompositeSolverStrategy <: AbstractSolverStrategy end

"""
    is_mixed_stiffness_problem(analysis::SystemAnalysis) -> Bool

Determine if the problem has mixed SL_STIFF/non-SL_STIFF characteristics that would 
benefit from IMEX or hybrid methods.
"""
function is_mixed_stiffness_problem(analysis::SystemAnalysis)
    # Check for moderate stiffness with multiscale behavior
    stiffness = classify_stiffness(analysis)
    has_multiscale = has_multiscale_behavior(analysis)
    
    # Mixed problems typically have moderate stiffness ratios but clear timescale separation
    if stiffness in [SL_MILDLY_STIFF, SL_STIFF] && has_multiscale
        return true
    end
    
    # Check coupling strength - mixed problems often have weakly coupled components
    if analysis.coupling_strength < 0.5 && stiffness != SL_NON_STIFF
        return true
    end
    
    # Systems with sparse structure and moderate stiffness often benefit from splitting
    if analysis.is_sparse && stiffness in [SL_MILDLY_STIFF, SL_STIFF]
        return true
    end
    
    return false
end

"""
    estimate_explicit_fraction(analysis::SystemAnalysis) -> Float64

Estimate what fraction of the system dynamics can be handled explicitly.
"""
function estimate_explicit_fraction(analysis::SystemAnalysis)
    # Base estimate from stiffness ratio
    ratio = analysis.stiffness_ratio
    
    if ratio < 10
        return 0.9  # Mostly explicit
    elseif ratio < 50
        return 0.7
    elseif ratio < 200
        return 0.5
    elseif ratio < 1000
        return 0.3
    else
        return 0.1  # Mostly implicit
    end
end

"""
    get_composite_recommendations(analysis::SystemAnalysis) -> Vector{AlgorithmRecommendation}

Get algorithm recommendations for composite (IMEX/hybrid) methods.
"""
function get_composite_recommendations(analysis::SystemAnalysis; rtol::Float64=1e-6, prefer_memory::Bool=false, prefer_stability::Bool=true)
    stiffness = classify_stiffness(analysis)
    sys_size = classify_system_size(analysis)
    is_sparse = requires_sparse_handling(analysis)
    is_mixed = is_mixed_stiffness_problem(analysis)
    explicit_frac = estimate_explicit_fraction(analysis)
    
    recommendations = AlgorithmRecommendation[]
    
    # High-order IMEX methods for well-separated timescales
    if is_mixed && explicit_frac > 0.4
        
        # KenCarp methods - High-order ESDIRK for stiff/mixed problems
        if sys_size != SS_LARGE_SYSTEM
            push!(recommendations, AlgorithmRecommendation(
                KenCarp4(), 9.2, COMPOSITE,
                min_accuracy=1e-12,
                max_accuracy=1e-4,
                memory_efficiency=0.75,
                computational_cost=0.6,
                stability_score=0.9,
                stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
                system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
                description="4th order ESDIRK method, excellent for mixed stiff problems",
                references=["Kennedy & Carpenter (2003)"]
            ))
            
            push!(recommendations, AlgorithmRecommendation(
                KenCarp5(), 8.9, COMPOSITE,
                min_accuracy=1e-14,
                max_accuracy=1e-5,
                memory_efficiency=0.7,
                computational_cost=0.7,
                stability_score=0.95,
                stiffness_range=(SL_STIFF, SL_VERY_STIFF),
                system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
                description="5th order ESDIRK method for high accuracy"
            ))
        end
        
        # IMEX multistep methods
        push!(recommendations, AlgorithmRecommendation(
            CNAB2(), 8.3, COMPOSITE,
            min_accuracy=1e-8,
            max_accuracy=1e-2,
            memory_efficiency=0.9,
            computational_cost=0.4,
            stability_score=0.75,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Crank-Nicolson Adams-Bashforth IMEX method, memory efficient"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            CNLF2(), 8.0, COMPOSITE,
            min_accuracy=1e-7,
            max_accuracy=1e-2,
            memory_efficiency=0.9,
            computational_cost=0.35,
            stability_score=0.7,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Crank-Nicolson Leapfrog IMEX method"
        ))
    end
    
    # Operator splitting methods for systems with clear physical separation
    if is_mixed && analysis.is_sparse && explicit_frac > 0.3
        
        push!(recommendations, AlgorithmRecommendation(
            SplitEuler(), 7.8, COMPOSITE,
            min_accuracy=1e-6,
            max_accuracy=1e-1,
            memory_efficiency=0.95,
            computational_cost=0.3,
            stability_score=0.7,
            stiffness_range=(SL_MILDLY_STIFF, SL_VERY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            handles_sparse=true,
            description="First-order operator splitting, very memory efficient for sparse systems"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            KenCarp3(), 8.7, COMPOSITE,
            min_accuracy=1e-10,
            max_accuracy=1e-3,
            memory_efficiency=0.8,
            computational_cost=0.5,
            stability_score=0.85,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="3rd order ESDIRK method with IMEX capabilities"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            KenCarp4(), 8.8, COMPOSITE,
            min_accuracy=1e-12,
            max_accuracy=1e-3,
            memory_efficiency=0.75,
            computational_cost=0.55,
            stability_score=0.9,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="4th order ESDIRK method, excellent for mixed SL_STIFF problems"
        ))
    end
    
    # Exponential integrators for linear SL_STIFF parts
    if stiffness in [SL_STIFF, SL_VERY_STIFF] && sys_size != SS_LARGE_SYSTEM && analysis.coupling_strength < 0.7
        
        push!(recommendations, AlgorithmRecommendation(
            ETDRK4(), 8.6, COMPOSITE,
            min_accuracy=1e-10,
            max_accuracy=1e-3,
            memory_efficiency=0.7,
            computational_cost=0.7,
            stability_score=0.9,
            stiffness_range=(SL_STIFF, SL_VERY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="4th order exponential time differencing, excellent for semilinear problems",
            references=["Cox & Matthews (2002)"]
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            ETDRK3(), 8.3, COMPOSITE,
            min_accuracy=1e-8,
            max_accuracy=1e-2,
            memory_efficiency=0.75,
            computational_cost=0.65,
            stability_score=0.85,
            stiffness_range=(SL_STIFF, SL_VERY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="3rd order exponential time differencing"
        ))
    end
    
    # Rosenbrock-W methods (linearly implicit IMEX-like behavior)
    if stiffness in [SL_MILDLY_STIFF, SL_STIFF] && is_mixed
        
        push!(recommendations, AlgorithmRecommendation(
            RosenbrockW6S4OS(), 8.4, COMPOSITE,
            min_accuracy=1e-12,
            max_accuracy=1e-4,
            memory_efficiency=0.8,
            computational_cost=0.6,
            stability_score=0.85,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="4th order Rosenbrock-W method with excellent stability"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            Rodas5P(), 8.9, COMPOSITE,  # Also good for composite problems
            min_accuracy=1e-12,
            max_accuracy=1e-4,
            memory_efficiency=0.85,
            computational_cost=0.5,
            stability_score=0.9,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="5th order Rosenbrock method, excellent for mixed SL_STIFF/non-SL_STIFF problems"
        ))
    end
    
    # Stabilized explicit methods for mildly SL_STIFF problems
    if stiffness == SL_MILDLY_STIFF && explicit_frac > 0.6
        
        push!(recommendations, AlgorithmRecommendation(
            ROCK2(), 8.1, COMPOSITE,
            min_accuracy=1e-8,
            max_accuracy=1e-2,
            memory_efficiency=0.9,
            computational_cost=0.4,
            stability_score=0.8,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Stabilized explicit method for mildly SL_STIFF problems"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            ROCK4(), 8.2, COMPOSITE,
            min_accuracy=1e-10,
            max_accuracy=1e-2,
            memory_efficiency=0.85,
            computational_cost=0.5,
            stability_score=0.85,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="4th order stabilized explicit method"
        ))
    end
    
    # For large sparse systems with mixed character
    if is_sparse && sys_size == SS_LARGE_SYSTEM && is_mixed
        
        push!(recommendations, AlgorithmRecommendation(
            TRBDF2(), 8.3, COMPOSITE,
            min_accuracy=1e-8,
            max_accuracy=1e-2,
            memory_efficiency=0.8,
            computational_cost=0.6,
            stability_score=0.8,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM),
            handles_sparse=true,
            description="TR-BDF2 method, good for large sparse mixed problems"
        ))
        
        # SUNDIALS ARKODE for large IMEX problems
        push!(recommendations, AlgorithmRecommendation(
            ARKODE(), 8.7, COMPOSITE,
            min_accuracy=1e-10,
            max_accuracy=1e-3,
            memory_efficiency=0.8,
            computational_cost=0.65,
            stability_score=0.9,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM),
            handles_sparse=true,
            description="SUNDIALS ARKODE for large-scale IMEX problems",
            references=["Reynolds et al. (2018)"]
        ))
    end
    
    # Multirate methods for problems with clear timescale separation
    if has_multiscale_behavior(analysis) && sys_size != SS_LARGE_SYSTEM
        
    end
    
    # Hybrid explicit-implicit methods
    if explicit_frac > 0.5 && stiffness in [SL_MILDLY_STIFF, SL_STIFF]
        
        push!(recommendations, AlgorithmRecommendation(
            SSPRK22(), 7.5, COMPOSITE,
            min_accuracy=1e-6,
            max_accuracy=1e-2,
            memory_efficiency=0.95,
            computational_cost=0.3,
            stability_score=0.7,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Strong stability preserving method for mixed problems with dominant explicit part"
        ))
    end
    
    return recommendations
end

"""
    analyze_splitting_potential(analysis::SystemAnalysis) -> Dict{Symbol, Float64}

Analyze how well-suited the system is for operator splitting approaches.
"""
function analyze_splitting_potential(analysis::SystemAnalysis)
    scores = Dict{Symbol, Float64}()
    
    # Additive splitting potential (f = f_fast + f_slow)
    if has_multiscale_behavior(analysis)
        timescale_ratio = maximum(abs.(analysis.timescales)) / minimum(abs.(analysis.timescales[analysis.timescales .!= 0]))
        scores[:additive_splitting] = min(timescale_ratio / 1000, 1.0)
    else
        scores[:additive_splitting] = 0.1
    end
    
    # Multiplicative splitting potential (based on system structure)
    if analysis.is_sparse && analysis.coupling_strength < 0.6
        scores[:multiplicative_splitting] = 0.8
    else
        scores[:multiplicative_splitting] = 0.3
    end
    
    # Dimensional splitting potential (for PDE-like systems)
    if analysis.system_size > 100 && analysis.is_sparse
        scores[:dimensional_splitting] = 0.7
    else
        scores[:dimensional_splitting] = 0.2
    end
    
    return scores
end

"""
    configure_composite_solver(alg, analysis::SystemAnalysis; rtol::Float64=1e-6, atol::Float64=1e-9)

Configure solver-specific options for composite methods.
"""
function configure_composite_solver(alg, analysis::SystemAnalysis; rtol::Float64=1e-6, atol::Float64=1e-9)
    base_options = Dict{Symbol, Any}(
        :reltol => rtol,
        :abstol => atol,
        :maxiters => 1000000
    )
    
    # IMEX-specific configurations
    if string(typeof(alg)) |> x -> occursin("ARK", x) || occursin("IMEX", x)
        # More aggressive step size control for IMEX methods
        base_options[:qmax] = 3.0
        base_options[:qmin] = 0.3
        
        # Adjust for explicit fraction
        explicit_frac = estimate_explicit_fraction(analysis)
        if explicit_frac > 0.7
            base_options[:dtmax] = 0.5  # Allow larger steps when mostly explicit
        elseif explicit_frac < 0.3
            base_options[:dtmax] = 0.1  # Limit steps when mostly implicit
        end
    end
    
    # Exponential integrator configurations
    if string(typeof(alg)) |> x -> occursin("ETDRK", x)
        base_options[:krylov] = true
        base_options[:m] = min(30, analysis.system_size ÷ 10)  # Krylov subspace size
    end
    
    # Operator splitting configurations
    if string(typeof(alg)) |> x -> occursin("Split", x)
        splitting_scores = analyze_splitting_potential(analysis)
        if splitting_scores[:additive_splitting] > 0.5
            base_options[:split_type] = :additive
        else
            base_options[:split_type] = :multiplicative
        end
    end
    
    # Sparse system optimizations
    if requires_sparse_handling(analysis)
        if hasfield(typeof(alg), :linsolve)
            if analysis.system_size > 1000
                base_options[:linsolve] = :GMRES
            else
                base_options[:linsolve] = :UMFPACK
            end
        end
    end
    
    # Multiscale-specific adjustments
    if has_multiscale_behavior(analysis)
        base_options[:adaptive] = true
        base_options[:beta2] = 0.08  # More conservative for multiscale
        base_options[:dtmin] = 1e-12  # Allow very small steps
    end
    
    return base_options
end

"""
    recommend_composite_solver(analysis::SystemAnalysis;
                               rtol::Float64=1e-6,
                               prefer_memory::Bool=false,
                               prefer_stability::Bool=false) -> Tuple{Any, Dict}

Get the single best composite solver recommendation with configuration.
"""
function recommend_composite_solver(analysis::SystemAnalysis;
                                   rtol::Float64=1e-6,
                                   prefer_memory::Bool=false,
                                   prefer_stability::Bool=false)
    
    # Check if problem is suitable for composite methods
    if !is_mixed_stiffness_problem(analysis)
        @info "Problem doesn't appear to have mixed SL_STIFF/non-SL_STIFF character. Consider explicit or SL_STIFF solvers instead."
    end
    
    recommendations = get_composite_recommendations(analysis)
    
    # Filter applicable recommendations
    applicable = filter(rec -> is_applicable(rec, analysis, rtol), recommendations)
    
    if isempty(applicable)
        @warn "No applicable composite solvers found, falling back to ARK324L2SA2"
        best_rec = AlgorithmRecommendation(ARK324L2SA2(), 5.0, COMPOSITE)
    else
        # Compute adjusted priorities and sort
        priorities = [compute_adjusted_priority(rec, analysis;
                                               prefer_memory=prefer_memory,
                                               prefer_stability=prefer_stability)
                     for rec in applicable]
        
        best_idx = argmax(priorities)
        best_rec = applicable[best_idx]
    end
    
    # Configure the selected solver
    config = configure_composite_solver(best_rec.algorithm, analysis; rtol=rtol)
    
    return best_rec.algorithm, config
end

"""
    suggest_problem_splitting(analysis::SystemAnalysis) -> Dict{Symbol, Any}

Suggest how to split the problem for operator splitting methods.
"""
function suggest_problem_splitting(analysis::SystemAnalysis)
    suggestions = Dict{Symbol, Any}()
    
    splitting_scores = analyze_splitting_potential(analysis)
    
    if splitting_scores[:additive_splitting] > 0.6
        suggestions[:recommended_splitting] = :additive
        suggestions[:fast_variables] = Int[]  # Would need problem-specific analysis
        suggestions[:slow_variables] = Int[]  # Would need problem-specific analysis
        suggestions[:description] = "System shows good potential for additive splitting (f = f_fast + f_slow)"
    elseif splitting_scores[:multiplicative_splitting] > 0.6
        suggestions[:recommended_splitting] = :multiplicative
        suggestions[:description] = "System structure suggests multiplicative splitting might be effective"
    elseif splitting_scores[:dimensional_splitting] > 0.6
        suggestions[:recommended_splitting] = :dimensional
        suggestions[:description] = "Large sparse system may benefit from dimensional splitting"
    else
        suggestions[:recommended_splitting] = :none
        suggestions[:description] = "System doesn't show clear structure for operator splitting"
    end
    
    return suggestions
end

end # module CompositeSolvers

