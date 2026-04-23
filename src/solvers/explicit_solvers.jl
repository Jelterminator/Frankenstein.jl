module ExplicitSolvers

using DifferentialEquations
using OrdinaryDiffEq
using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, EXPLICIT

# Re-export key types/functions to Solvers if needed
export ExplicitSolverStrategy, get_explicit_recommendations, select_best_explicit

# Implementation

"""
    ExplicitSolverStrategy <: AbstractSolverStrategy

Strategy for selecting explicit methods suitable for non-SL_STIFF problems.
"""
struct ExplicitSolverStrategy <: AbstractSolverStrategy
    prefer_high_order::Bool
    prefer_memory_efficient::Bool
    max_system_size::Int
    
    ExplicitSolverStrategy(; prefer_high_order::Bool=false, 
                          prefer_memory_efficient::Bool=false,
                          max_system_size::Int=10000) = 
        new(prefer_high_order, prefer_memory_efficient, max_system_size)
end

#==============================================================================#
# Explicit Algorithm Catalog
#==============================================================================#

"""
    create_explicit_algorithms() -> Vector{AlgorithmRecommendation}

Create catalog of explicit algorithms with their characteristics.
"""
function create_explicit_algorithms()
    return [
        # High-performance general purpose explicit methods
        AlgorithmRecommendation(
            Tsit5, 9.5, EXPLICIT;
            min_accuracy=1e-12, max_accuracy=1e-3,
            memory_efficiency=0.9, computational_cost=0.3, stability_score=0.85,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Fifth-order Tsitouras method - best general explicit solver",
            references=["Tsitouras, C. (2011). Runge–Kutta pairs of order 5(4)"]
        ),
        
        AlgorithmRecommendation(
            DP5, 8.5, EXPLICIT;
            min_accuracy=1e-10, max_accuracy=1e-2, 
            memory_efficiency=0.85, computational_cost=0.35, stability_score=0.8,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Dormand-Prince 5(4) - classic and reliable",
            references=["Dormand & Prince (1980). A family of embedded Runge-Kutta formulae"]
        ),
        
        # High-order methods for demanding accuracy
        AlgorithmRecommendation(
            Vern7, 8.0, EXPLICIT;
            min_accuracy=1e-14, max_accuracy=1e-5,
            memory_efficiency=0.7, computational_cost=0.6, stability_score=0.9,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="7th order Verner method for high accuracy",
            references=["Verner, J.H. (1978). Explicit Runge-Kutta methods with estimates"]
        ),
        
        AlgorithmRecommendation(
            Vern8, 7.8, EXPLICIT;
            min_accuracy=1e-15, max_accuracy=1e-6,
            memory_efficiency=0.65, computational_cost=0.7, stability_score=0.9,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="8th order Verner method for very high accuracy"
        ),
        
        AlgorithmRecommendation(
            Vern9, 7.5, EXPLICIT;
            min_accuracy=1e-16, max_accuracy=1e-8,
            memory_efficiency=0.6, computational_cost=0.8, stability_score=0.95,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM),
            description="9th order Verner - highest accuracy explicit method"
        ),
        
        # Medium-order balanced methods
        AlgorithmRecommendation(
            Vern6, 8.2, EXPLICIT;
            min_accuracy=1e-12, max_accuracy=1e-4,
            memory_efficiency=0.8, computational_cost=0.45, stability_score=0.85,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="6th order Verner - good accuracy/efficiency balance"
        ),
        
        AlgorithmRecommendation(
            TanYam7, 7.0, EXPLICIT;
            min_accuracy=1e-12, max_accuracy=1e-5,
            memory_efficiency=0.75, computational_cost=0.55, stability_score=0.8,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Tanaka-Yamashita 7th order method"
        ),
        
        # Lower-order efficient methods
        AlgorithmRecommendation(
            BS3, 7.5, EXPLICIT;
            min_accuracy=1e-8, max_accuracy=1e-1,
            memory_efficiency=0.95, computational_cost=0.2, stability_score=0.7,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Bogacki-Shampine 3(2) - fast and memory efficient",
            references=["Bogacki & Shampine (1989). A 3(2) pair of Runge-Kutta formulas"]
        ),
        
        AlgorithmRecommendation(
            BS5, 7.0, EXPLICIT;
            min_accuracy=1e-10, max_accuracy=1e-3,
            memory_efficiency=0.85, computational_cost=0.4, stability_score=0.75,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Bogacki-Shampine 5(4) method"
        ),
        
        # Specialized explicit methods
        AlgorithmRecommendation(
            Feagin10, 6.5, EXPLICIT;
            min_accuracy=1e-16, max_accuracy=1e-10,
            memory_efficiency=0.5, computational_cost=0.9, stability_score=0.95,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_SMALL_SYSTEM),
            description="Feagin 10th order - for extreme accuracy requirements"
        ),
        
        AlgorithmRecommendation(
            Feagin12, 6.0, EXPLICIT;
            min_accuracy=1e-17, max_accuracy=1e-12,
            memory_efficiency=0.45, computational_cost=0.95, stability_score=0.95,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            system_size_range=(SS_SMALL_SYSTEM, SS_SMALL_SYSTEM),
            description="Feagin 12th order - maximum accuracy explicit method"
        ),
        
        # Fast, low-accuracy methods
        AlgorithmRecommendation(
            Euler, 6.0, EXPLICIT;
            min_accuracy=1e-3, max_accuracy=1e0,
            memory_efficiency=0.99, computational_cost=0.1, stability_score=0.4,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Explicit Euler - fastest possible, very low accuracy"
        ),
        
        AlgorithmRecommendation(
            Midpoint, 6.2, EXPLICIT;
            min_accuracy=1e-4, max_accuracy=1e-1,
            memory_efficiency=0.98, computational_cost=0.15, stability_score=0.5,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Explicit midpoint method - simple 2nd order"
        ),
        
        AlgorithmRecommendation(
            Heun, 6.5, EXPLICIT;
            min_accuracy=1e-5, max_accuracy=1e-2,
            memory_efficiency=0.95, computational_cost=0.2, stability_score=0.6,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Heun's method - improved Euler"
        ),
        
        # Step size control variants
        AlgorithmRecommendation(
            OwrenZen3, 7.2, EXPLICIT;
            min_accuracy=1e-8, max_accuracy=1e-2,
            memory_efficiency=0.9, computational_cost=0.25, stability_score=0.75,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Owren-Zennaro optimized 3rd order method"
        ),
        
        AlgorithmRecommendation(
            OwrenZen4, 7.5, EXPLICIT;
            min_accuracy=1e-10, max_accuracy=1e-3,
            memory_efficiency=0.85, computational_cost=0.3, stability_score=0.8,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Owren-Zennaro optimized 4th order method"
        ),
        
        AlgorithmRecommendation(
            OwrenZen5, 7.8, EXPLICIT;
            min_accuracy=1e-12, max_accuracy=1e-4,
            memory_efficiency=0.8, computational_cost=0.35, stability_score=0.82,
            stiffness_range=(SL_NON_STIFF, SL_NON_STIFF),
            description="Owren-Zennaro optimized 5th order method"
        )

    ]
end

#==============================================================================#
# Selection Logic for Explicit Methods
#==============================================================================#

"""
    get_explicit_recommendations(analysis::SystemAnalysis; 
                                rtol::Float64=1e-6,
                                strategy::ExplicitSolverStrategy=ExplicitSolverStrategy()) 
                                -> Vector{AlgorithmRecommendation}

Get prioritized explicit algorithm recommendations based on system analysis.
"""
function get_explicit_recommendations(analysis::SystemAnalysis;
                                    rtol::Float64=1e-6,
                                    prefer_memory::Bool=false,
                                    prefer_stability::Bool=true,
                                    strategy::ExplicitSolverStrategy=ExplicitSolverStrategy())
    
    # Only recommend explicit methods for non-SL_STIFF problems
    stiffness = classify_stiffness(analysis)
    if stiffness != SL_NON_STIFF
        return AlgorithmRecommendation[]
    end
    
    # Check system size constraints
    if analysis.system_size > strategy.max_system_size
        return AlgorithmRecommendation[]
    end
    
    algorithms = create_explicit_algorithms()
    
    # Filter by applicability
    applicable = filter(alg -> is_applicable(alg, analysis, rtol), algorithms)
    
    # Adjust priorities based on strategy and problem characteristics
    adjusted = map(applicable) do rec
        priority = compute_adjusted_priority(rec, analysis; 
                                           prefer_memory=strategy.prefer_memory_efficient)
        
        # Strategy-specific adjustments
        if strategy.prefer_high_order && occursin("order", rec.description)
            # Extract order from description and boost high-order methods
            if occursin("9", rec.description) || occursin("10", rec.description) || 
               occursin("12", rec.description)
                priority += 1.5
            elseif occursin("7", rec.description) || occursin("8", rec.description)
                priority += 1.0
            elseif occursin("6", rec.description)
                priority += 0.5
            end
        end
        
        # Prefer methods with better stability for larger systems
        sys_size = classify_system_size(analysis)
        if sys_size == SS_LARGE_SYSTEM
            priority += rec.stability_score * 0.5
        end
        
        AlgorithmRecommendation(
            rec.algorithm, priority, rec.category,
            min_accuracy=rec.min_accuracy, max_accuracy=rec.max_accuracy,
            memory_efficiency=rec.memory_efficiency, computational_cost=rec.computational_cost,
            stability_score=rec.stability_score, stiffness_range=rec.stiffness_range,
            system_size_range=rec.system_size_range, handles_sparse=rec.handles_sparse,
            handles_mass_matrix=rec.handles_mass_matrix, supports_events=rec.supports_events,
            description=rec.description, references=rec.references
        )
    end
    
    # Sort by priority and return
    sort(adjusted, by=x->x.priority, rev=true)
end

"""
    select_best_explicit(analysis::SystemAnalysis; kwargs...) -> Union{Nothing, Any}

Select the single best explicit algorithm, or nothing if explicit methods are not suitable.
"""
function select_best_explicit(analysis::SystemAnalysis; kwargs...)
    recommendations = get_explicit_recommendations(analysis; kwargs...)
    
    if isempty(recommendations)
        return nothing
    end
    
    return recommendations[1].algorithm
end

end # module ExplicitSolvers

