module FCore

using SciMLBase

# Include fundamental types and enums
include("types.jl")

# Export public API for other modules
export FrankensteinSolver,
       SystemAnalysis, StepInfo,
       SolverConfiguration,
       PerformanceProfile,
       AdaptationState,
       AbstractMonsterSolver,
       AbstractADBackend,
       AbstractSolverStrategy,
       AbstractPreconditioner,
       AbstractSplittingMethod,
       AbstractAdaptationStrategy,
       AbstractPerformanceMonitor

# Exported API
export # Categories and Levels
       SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel,
       EXPLICIT, STABILIZED_EXPLICIT, STIFF, COMPOSITE, MULTISCALE, SPARSE, ADAPTIVE, PARALLEL, SPECIALTY,
       SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF,
       SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, SS_EXTREME_SYSTEM,
       LOW_ACCURACY, STANDARD_ACCURACY, HIGH_ACCURACY,
       AlgorithmRecommendation,
       is_applicable, compute_adjusted_priority,
       classify_stiffness, classify_system_size, classify_accuracy_level,
       requires_sparse_handling, is_well_conditioned, has_multiscale_behavior,
       
       # Hysteresis Borders
       BORDER_STIFF_UP, BORDER_STIFF_DOWN

#==============================================================================#
# Constants & Hysteresis Borders
#==============================================================================#

const BORDER_STIFF_UP = 1000.0   # Non-Stiff -> Stiff
const BORDER_STIFF_DOWN = 100.0  # Stiff -> Non-Stiff

#==============================================================================#
# Abstract Types
#==============================================================================#

"""
    AbstractMonsterSolver <: SciMLBase.SciMLAlgorithm

Top-level abstract type for all Frankenstein solvers. It subtypes `SciMLAlgorithm`
to integrate with DifferentialEquations.jl.
"""
abstract type AbstractMonsterSolver <: SciMLBase.SciMLAlgorithm end

"""
    AbstractSolverStrategy
"""
abstract type AbstractSolverStrategy end

"""
    AbstractPreconditioner
"""
abstract type AbstractPreconditioner end

"""
    AbstractSplittingMethod
"""
abstract type AbstractSplittingMethod end

"""
    AbstractAdaptationStrategy
"""
abstract type AbstractAdaptationStrategy end

"""
    AbstractPerformanceMonitor
"""
abstract type AbstractPerformanceMonitor end

#==============================================================================#
# Concrete FCore Types
#==============================================================================#

"""
    FrankensteinSolver(...)
    
The Monster Solver Algorithm™ 🧟.
"""
mutable struct FrankensteinSolver <: AbstractMonsterSolver
    configuration::Any # Will hold SolverConfiguration
    analysis::Any      # Will hold SystemAnalysis
    adaptation::Any    # hold AdaptationState
    disabled_backends::Dict{String, Int} # Backend Name => disabled until step N
    recovery_attempts::Int
    original_f::Any    # The unwrapped user function
end

# Constructor
function FrankensteinSolver(; kwargs...)
    return FrankensteinSolver(nothing, nothing, nothing, Dict{String, Int}(), 0, nothing)
end

"""
    SolverConfiguration{Alg, AD, LS}
"""
struct SolverConfiguration{Alg, AD, LS}
    solver::Alg
    ad_backend::AD
    linear_solver::LS
end

"""
    PerformanceProfile{T}
"""
mutable struct PerformanceProfile{T}
    solve_time_s::T
    num_steps::Int
    num_f_evals::Int
    num_jac_evals::Int
    num_linsolves::Int
    num_step_rejects::Int
end

function PerformanceProfile{T}() where T
    return PerformanceProfile{T}(T(0), 0, 0, 0, 0, 0)
end

"""
    AdaptationState
"""
mutable struct AdaptationState
    current_strategy::AbstractAdaptationStrategy
    history::Vector{Any}
    AdaptationState(strategy::AbstractAdaptationStrategy) = new(strategy, Any[])
end

end # module FCore
