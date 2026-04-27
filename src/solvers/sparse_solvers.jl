# sparse_solvers.jl - Sparse system solver strategies

module SparseSolvers

using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, SS_EXTREME_SYSTEM, SPARSE, STABILIZED_EXPLICIT
using OrdinaryDiffEq
using LinearSolve
using SparseArrays

#==============================================================================#
# Sparse Solver Strategy Definition
#==============================================================================#

"""
    SparseSolverStrategy

Encapsulates logic for recommending ODE solvers optimized for sparse Jacobians or systems.
"""
struct SparseSolverStrategy
    catalogue::Vector{AlgorithmRecommendation}
end

#==============================================================================#
# Solver Catalogue for Sparse Systems
#==============================================================================#

function build_sparse_solver_catalogue()
    AlgorithmRecommendation[
        AlgorithmRecommendation(TRBDF2, 9.2, SPARSE;
            description = "TRBDF2 with sparse linear solver using KLUFactorization.",
            handles_sparse = true,
            stability_score = 0.9,
            memory_efficiency = 0.7,
            computational_cost = 0.4,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(QNDF, 9.0, SPARSE;
            description = "QNDF with sparse direct solver. Good alternative to FBDF.",
            handles_sparse = true,
            stability_score = 0.85,
            computational_cost = 0.5),

        AlgorithmRecommendation(FBDF, 8.9, SPARSE;
            description = "FBDF with sparse direct linear solver. Highly robust for large sparse systems.",
            handles_sparse = true,
            stability_score = 0.9,
            computational_cost = 0.5),
        
        AlgorithmRecommendation(KenCarp4, 8.8, SPARSE;
            description = "KenCarp4 IMEX scheme with sparse linear solver support.",
            handles_sparse = true,
            stability_score = 0.9,
            computational_cost = 0.5,
            references = ["Kennedy & Carpenter (2003)"]),

        AlgorithmRecommendation(Rosenbrock23, 7.5, SPARSE;
            description = "Rosenbrock23 method with sparse KLU solver.",
            handles_sparse = true,
            stability_score = 0.85,
            computational_cost = 0.5,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),
    ]
end


#==============================================================================#
# Recommendation Function
#==============================================================================#

"""
    get_sparse_recommendations(analysis::SystemAnalysis; rtol=1e-6, prefer_memory=false, prefer_stability=true)

Return a sorted list of sparse-optimized algorithms suitable for the given system.
"""
function get_sparse_recommendations(analysis::SystemAnalysis;
                                    rtol::Float64=1e-6,
                                    prefer_memory::Bool=false,
                                    prefer_stability::Bool=true)

    catalogue = build_sparse_solver_catalogue()

    filtered = filter(rec -> is_applicable(rec, analysis, rtol), catalogue)

    scored = sort(filtered; by = rec -> -compute_adjusted_priority(rec, analysis;
                                        prefer_memory=prefer_memory,
                                        prefer_stability=prefer_stability))

    return scored
end

export SparseSolverStrategy, get_sparse_recommendations

end # module


