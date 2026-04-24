module FCore

using SciMLBase

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
       EXPLICIT, STIFF, COMPOSITE, MULTISCALE, SPARSE, ADAPTIVE, PARALLEL, SPECIALTY,
       SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF,
       SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM,
       LOW_ACCURACY, STANDARD_ACCURACY, HIGH_ACCURACY,
       AlgorithmRecommendation,
       is_applicable, compute_adjusted_priority,
       classify_stiffness, classify_system_size, classify_accuracy_level,
       requires_sparse_handling, is_well_conditioned, has_multiscale_behavior

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
end

# Constructor
function FrankensteinSolver(; kwargs...)
    return FrankensteinSolver(nothing, nothing, nothing, Dict{String, Int}(), 0)
end

# (solve dispatch moved to MonsterSolver.jl)

"""
    SystemAnalysis{T}
"""
mutable struct SystemAnalysis{T}
    stiffness_ratio::T
    is_stiff::Bool
    sparsity_pattern::Any
    timescales::Vector{T}
    coupling_strength::T
    condition_number::T
    system_size::Int
    is_sparse::Bool
    jacobian::Any
    stable_count::Int
    last_update_step::Int
    current_step::Int
    last_norm_du::T
    last_jacobian_update::Int
    history::Vector{Any}
end

function SystemAnalysis{T}() where T
    return SystemAnalysis{T}(
        T(NaN), false, nothing, T[], T(NaN), T(NaN), 
        0, false, nothing, 0, 0, 0, T(0), 0, Any[]
    )
end

function SystemAnalysis()
    return SystemAnalysis{Float64}()
end

"""
    StepInfo{T, P}
"""
struct StepInfo{T, P}
    u::Vector{T}
    du::Vector{T}
    dt::T
    dt_prev::T
    rejects::Int
    nsteps::Int
    t::T
    p::P
    prob::SciMLBase.ODEProblem
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

# Include sub-files
include("types.jl")

end # module FCore
