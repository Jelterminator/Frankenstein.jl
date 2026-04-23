# sparse_solvers.jl - Sparse system solver strategies

module SparseSolvers

using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, SPARSE
using OrdinaryDiffEq
using Sundials
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
        AlgorithmRecommendation(FBDF, 9.5, SPARSE;
            description = "FBDF with sparse direct linear solver. Highly robust for large sparse systems.",
            handles_sparse = true,
            stability_score = 0.9,
            computational_cost = 0.5),

        AlgorithmRecommendation(QNDF, 9.3, SPARSE;
            description = "QNDF with sparse direct solver. Good alternative to FBDF.",
            handles_sparse = true,
            stability_score = 0.85,
            computational_cost = 0.5),

        AlgorithmRecommendation(CVODE_BDF, 8.0, SPARSE;
            description = "SUNDIALS CVODE_BDF with KLU sparse direct solver.",
            handles_sparse = true,
            handles_mass_matrix = true,
            is_sundials = true,
            stability_score = 0.95,
            computational_cost = 0.6,
            references = ["https://computing.llnl.gov/projects/sundials"]),

        AlgorithmRecommendation(CVODE_BDF, 8.8, SPARSE;
            description = "SUNDIALS CVODE_BDF with GMRES iterative solver. Good for large sparse systems where direct solvers are too costly.",
            handles_sparse = true,
            handles_mass_matrix = true,
            is_sundials = true,
            stability_score = 0.9,
            computational_cost = 0.7),
        
        AlgorithmRecommendation(TRBDF2, 8.5, SPARSE;
            description = "TRBDF2 with sparse linear solver using KLUFactorization.",
            handles_sparse = true,
            stability_score = 0.9,
            memory_efficiency = 0.7,
            computational_cost = 0.4,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(Rosenbrock23, 7.5, SPARSE;
            description = "Rosenbrock23 method with sparse KLU solver.",
            handles_sparse = true,
            stability_score = 0.85,
            computational_cost = 0.5,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(KenCarp4, 7.0, SPARSE;
            description = "KenCarp4 IMEX scheme with sparse linear solver support.",
            handles_sparse = true,
            stability_score = 0.9,
            computational_cost = 0.5,
            references = ["Kennedy & Carpenter (2003)"]),
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


