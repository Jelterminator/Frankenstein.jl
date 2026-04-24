# Frankenstein.jl Solver Strategies Module Structure

"""
Solvers — automatic solver selection for ODE systems

1. Build up a `SystemAnalysis`
2. Call `select_best_algorithm(analysis; rtol, abstol, prefs…)`
3. Pass the returned `AlgorithmRecommendation` into `create_solver_configuration`
4. Use the resulting config in `solve(prob, config.algorithm; reltol=config.reltol, …)`
"""

module Solvers

using DifferentialEquations
using OrdinaryDiffEq
using Sundials
using LinearSolve
using SparseArrays
using ..FCore: FCore, SystemAnalysis, AbstractSolverStrategy
using ..Backends: Backends, PrecomputedSparsityDetector

# Re-export enums from FCore
using ..FCore: SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel,
             SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF,
             SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM,
             AlgorithmRecommendation,
             classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior,
             is_applicable, compute_adjusted_priority

export SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel,
       SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF,
       SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM,
       AlgorithmRecommendation,
       classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior,
       is_applicable, compute_adjusted_priority

# Include all solver strategy modules
include("explicit_solvers.jl")
using .ExplicitSolvers

include("stiff_solvers.jl")
using .StiffSolvers

include("composite_solvers.jl")
using .CompositeSolvers

include("multiscale_solvers.jl")
using .MultiscaleSolvers

include("sparse_solvers.jl")
using .SparseSolvers

include("adaptive_solvers.jl")
using .AdaptiveSolvers

include("parallel_solvers.jl")
using .ParallelSolvers

include("specialty_solvers.jl")
using .SpecialtySolvers

include("algorithm_selector.jl")
using .AlgorithmSelector

# Re-export all public interfaces
export AlgorithmRecommendation,
       SolverCategory,
       AlgorithmCatalogue,
       
       # From explicit_solvers.jl
       ExplicitSolverStrategy,
       get_explicit_recommendations,
       
       # From stiff_solvers.jl  
       StiffSolverStrategy,
       get_stiff_recommendations,
       
       # From composite_solvers.jl
       CompositeSolverStrategy,
       get_composite_recommendations,
       
       # From multiscale_solvers.jl
       MultiscaleSolverStrategy,
       get_multiscale_recommendations,
       
       # From sparse_solvers.jl
       SparseSolverStrategy,
       get_sparse_recommendations,
       
       # From adaptive_solvers.jl
       AdaptiveSolverStrategy,
       get_adaptive_recommendations,
       
       # From parallel_solvers.jl
       ParallelSolverStrategy,
       get_parallel_recommendations,
       
       # From specialty_solvers.jl
       SpecialtySolverStrategy,
       get_specialty_recommendations,
       
       # From algorithm_selector.jl
       select_algorithm,
       get_all_recommendations,
       create_solver_configuration

# Main interface function that delegates to the selector
"""
    select_best_algorithm(analysis::SystemAnalysis; kwargs...) 

Select the best algorithm for the given system analysis by consulting all solver strategy modules.
"""
function select_best_algorithm(analysis::SystemAnalysis; kwargs...)
    return select_algorithm(analysis; kwargs...)
end

end # module Solvers
