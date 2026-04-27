module StiffSolvers

using DifferentialEquations
using OrdinaryDiffEq
using LinearSolve
using SparseArrays
using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, SS_EXTREME_SYSTEM, STIFF, STABILIZED_EXPLICIT

# Re-export key types/functions to Solvers if needed
export StiffSolverStrategy, get_stiff_recommendations, get_linear_solver_recommendation,
       configure_stiff_solver, recommend_stiff_solver

# Implementation
# SL_STIFF Solver Strategy Implementation
#==============================================================================#

struct StiffSolverStrategy <: AbstractSolverStrategy end

"""
    get_stiff_recommendations(analysis::SystemAnalysis) -> Vector{AlgorithmRecommendation}

Get algorithm recommendations specifically for SL_STIFF problems, ordered by suitability.
"""
function get_stiff_recommendations(analysis::SystemAnalysis; rtol::Float64=1e-6, prefer_memory::Bool=false, prefer_stability::Bool=true)
    stiffness = classify_stiffness(analysis)
    sys_size = classify_system_size(analysis)
    is_sparse = requires_sparse_handling(analysis)
    is_well_cond = is_well_conditioned(analysis)
    
    recommendations = AlgorithmRecommendation[]
    
    # High-performance Rosenbrock methods (best for moderately SL_STIFF problems)
    if stiffness in [SL_MILDLY_STIFF, SL_STIFF]
        if sys_size == SS_SMALL_SYSTEM
            push!(recommendations, AlgorithmRecommendation(
                Rodas5P, 9.5, STIFF,
                min_accuracy=1e-14,
                max_accuracy=1e-4,
                memory_efficiency=0.9,
                computational_cost=0.4,
                stability_score=0.95,
                stiffness_range=(SL_MILDLY_STIFF, SL_VERY_STIFF),
                system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
                handles_sparse=false,
                supports_events=true,
                description="5th order Rosenbrock method with excellent stability and efficiency for small-medium SL_STIFF systems",
                references=["Steinebach & Orden (2006)"]
            ))
            
            push!(recommendations, AlgorithmRecommendation(
                Rodas5, 9.2, STIFF,
                min_accuracy=1e-12,
                max_accuracy=1e-3,
                memory_efficiency=0.85,
                computational_cost=0.45,
                stability_score=0.9,
                stiffness_range=(SL_MILDLY_STIFF, SL_VERY_STIFF),
                system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
                description="Robust 5th order Rosenbrock method with good performance characteristics"
            ))
        end
        
        push!(recommendations, AlgorithmRecommendation(
            Rodas4P, 8.8, STIFF,
            min_accuracy=1e-12,
            max_accuracy=1e-3,
            memory_efficiency=0.9,
            computational_cost=0.35,
            stability_score=0.9,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="4th order Rosenbrock method, very reliable for moderate stiffness"
        ))
    end
    
    # BDF methods for very SL_STIFF problems
    if stiffness in [STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF]
        push!(recommendations, AlgorithmRecommendation(
            QNDF, 9.3, STIFF,
            min_accuracy=1e-12,
            max_accuracy=1e-2,
            memory_efficiency=0.8,
            computational_cost=0.6,
            stability_score=0.95,
            stiffness_range=(SL_STIFF, SL_EXTREMELY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Quasi-constant step BDF method, excellent for very SL_STIFF problems"
        ))
        
        push!(recommendations, AlgorithmRecommendation(
            FBDF, 8.9, STIFF,
            min_accuracy=1e-10,
            max_accuracy=1e-2,
            memory_efficiency=0.75,
            computational_cost=0.7,
            stability_score=0.9,
            stiffness_range=(SL_STIFF, SL_EXTREMELY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Fixed-leading coefficient BDF, robust for SL_STIFF problems"
        ))
    end
    
    # Specialized methods for different conditions
    
    # High accuracy methods
    push!(recommendations, AlgorithmRecommendation(
        RadauIIA5, 8.7, STIFF,
        min_accuracy=1e-15,
        max_accuracy=1e-5,
        memory_efficiency=0.7,
        computational_cost=0.8,
        stability_score=0.98,
        stiffness_range=(SL_MILDLY_STIFF, SL_EXTREMELY_STIFF),
        system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
        description="5th order implicit Runge-Kutta method, excellent for high accuracy requirements"
    ))
    
    # For poorly conditioned systems
    if !is_well_cond
        push!(recommendations, AlgorithmRecommendation(
            Rodas4P2, 8.5, STIFF,
            min_accuracy=1e-10,
            max_accuracy=1e-3,
            memory_efficiency=0.85,
            computational_cost=0.4,
            stability_score=0.95,
            stiffness_range=(SL_MILDLY_STIFF, SL_VERY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            description="Robust Rosenbrock method with enhanced stability for ill-conditioned problems"
        ))
    end
    
    # Implicit Euler for extremely SL_STIFF or when robustness is paramount
    if stiffness == SL_EXTREMELY_STIFF || !is_well_cond
        push!(recommendations, AlgorithmRecommendation(
            ImplicitEuler, 7.5, STIFF,
            min_accuracy=1e-8,
            max_accuracy=1e-1,
            memory_efficiency=0.95,
            computational_cost=0.3,
            stability_score=1.0,
            stiffness_range=(SL_STIFF, SL_EXTREMELY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
            handles_sparse=true,
            description="1st order implicit method, maximum stability for extremely SL_STIFF problems"
        ))
    end
    
    # Trapezoidal rule for smooth problems
    if is_well_cond && sys_size != SS_LARGE_SYSTEM
        push!(recommendations, AlgorithmRecommendation(
            Trapezoid, 8.2, STIFF,
            min_accuracy=1e-10,
            max_accuracy=1e-2,
            memory_efficiency=0.9,
            computational_cost=0.35,
            stability_score=0.85,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="2nd order implicit trapezoidal rule, good for smooth SL_STIFF problems"
        ))
    end
    
    # Kvaerno methods for moderate stiffness
    if stiffness in [SL_MILDLY_STIFF, SL_STIFF] && sys_size != SS_LARGE_SYSTEM
        push!(recommendations, AlgorithmRecommendation(
            KenCarp4, 8.6, STIFF,
            min_accuracy=1e-12,
            max_accuracy=1e-3,
            memory_efficiency=0.8,
            computational_cost=0.5,
            stability_score=0.9,
            stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="4th order ESDIRK method with good stability properties"
        ))
    end
    
    # Sparse-optimized methods
    if is_sparse && sys_size == SS_LARGE_SYSTEM
        push!(recommendations, AlgorithmRecommendation(
            TRBDF2, 8.4, STIFF,
            min_accuracy=1e-10,
            max_accuracy=1e-2,
            memory_efficiency=0.85,
            computational_cost=0.6,
            stability_score=0.85,
            stiffness_range=(SL_STIFF, SL_VERY_STIFF),
            system_size_range=(SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM),
            handles_sparse=true,
            description="Trapezoidal-BDF2 method, good for large sparse systems"
        ))
    end
    
    # Low memory methods for resource-constrained environments
    push!(recommendations, AlgorithmRecommendation(
        Rosenbrock23, 7.8, STIFF,
        min_accuracy=1e-8,
        max_accuracy=1e-2,
        memory_efficiency=0.95,
        computational_cost=0.25,
        stability_score=0.8,
        stiffness_range=(SL_MILDLY_STIFF, SL_STIFF),
        system_size_range=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
        description="Low-order Rosenbrock method, very memory efficient"
    ))
    
    # For problems requiring high stability at the cost of accuracy
    if !is_well_cond || stiffness == SL_EXTREMELY_STIFF
        push!(recommendations, AlgorithmRecommendation(
            ImplicitMidpoint, 7.2, STIFF,
            min_accuracy=1e-6,
            max_accuracy=1e-2,
            memory_efficiency=0.9,
            computational_cost=0.4,
            stability_score=0.98,
            stiffness_range=(SL_STIFF, SL_EXTREMELY_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="2nd order implicit method with excellent stability properties"
        ))
    end
    
    applicable = filter(rec -> is_applicable(rec, analysis, rtol), recommendations)

    # Adjust priorities based on preferences
    adjusted = map(applicable) do rec
        priority = compute_adjusted_priority(rec, analysis; 
                                           prefer_memory=prefer_memory,
                                           prefer_stability=prefer_stability)
        
        # Maintain original metadata but update priority
        AlgorithmRecommendation(
            rec.algorithm, priority, rec.category;
            min_accuracy=rec.min_accuracy, max_accuracy=rec.max_accuracy,
            memory_efficiency=rec.memory_efficiency, computational_cost=rec.computational_cost,
            stability_score=rec.stability_score, stiffness_range=rec.stiffness_range,
            system_size_range=rec.system_size_range, handles_sparse=rec.handles_sparse,
            handles_mass_matrix=rec.handles_mass_matrix, supports_events=rec.supports_events,
            description=rec.description, references=rec.references
        )
    end

    return sort(adjusted; by = x -> -x.priority)
end

"""
    get_linear_solver_recommendation(analysis::SystemAnalysis) -> Symbol

Get the recommended linear solver for SL_STIFF problems based on system characteristics.
"""
function get_linear_solver_recommendation(analysis::SystemAnalysis)
    if requires_sparse_handling(analysis)
        if analysis.system_size > 1000
            return :GMRES
        else
            return :UMFPACK
        end
    else
        if analysis.system_size > 500
            return :LU
        else
            return :QR
        end
    end
end

"""
    configure_stiff_solver(alg, analysis::SystemAnalysis; rtol::Float64=1e-6, atol::Float64=1e-9)

Configure solver-specific options for SL_STIFF problems.
"""
function configure_stiff_solver(alg, analysis::SystemAnalysis; rtol::Float64=1e-6, atol::Float64=1e-9)
    base_options = Dict{Symbol, Any}(
        :reltol => rtol,
        :abstol => atol,
        :maxiters => 1000000
    )
    
    # Adjust based on stiffness level
    stiffness = classify_stiffness(analysis)
    if stiffness in [SL_VERY_STIFF, SL_EXTREMELY_STIFF]
        base_options[:dtmax] = 0.1  # Limit maximum step size
        base_options[:dtmin] = 1e-10  # Allow very small steps
    end
    
    # Sparse system optimizations
    if requires_sparse_handling(analysis)
        linear_solver = get_linear_solver_recommendation(analysis)
        if hasfield(typeof(alg), :linsolve)
            base_options[:linsolve] = linear_solver
        end
    end
    
    # Conditioning-based adjustments
    if !is_well_conditioned(analysis)
        base_options[:force_dtmin] = true
        base_options[:adaptive] = true
        base_options[:beta2] = 0.04  # More conservative step size control
    end
    
    return base_options
end

"""
    recommend_stiff_solver(analysis::SystemAnalysis; 
                          rtol::Float64=1e-6, 
                          prefer_memory::Bool=false,
                          prefer_stability::Bool=false) -> Tuple{Any, Dict}

Get the single best SL_STIFF solver recommendation with configuration.
"""
function recommend_stiff_solver(analysis::SystemAnalysis; 
                               rtol::Float64=1e-6,
                               prefer_memory::Bool=false,
                               prefer_stability::Bool=false)
    
    recommendations = get_stiff_recommendations(analysis)
    
    # Filter applicable recommendations
    applicable = filter(rec -> is_applicable(rec, analysis, rtol), recommendations)
    
    if isempty(applicable)
        @warn "No applicable SL_STIFF solvers found, falling back to Rodas4P"
        best_rec = AlgorithmRecommendation(Rodas4P, 5.0, STIFF)
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
    config = configure_stiff_solver(best_rec.algorithm, analysis; rtol=rtol)
    
    return best_rec.algorithm, config
end

end # module StiffSolvers

