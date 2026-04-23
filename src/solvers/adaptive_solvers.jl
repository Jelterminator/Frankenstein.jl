# adaptive_solvers.jl - Adaptive method switching and control-aware solvers

module AdaptiveSolvers

using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, ADAPTIVE
using OrdinaryDiffEq
using Sundials

#==============================================================================#
# Adaptive Solver Strategy Definition
#==============================================================================#

"""
    AdaptiveSolverStrategy

Encapsulates logic for recommending solvers with advanced adaptivity, such as time-step,
method order, and stiffness switching.
"""
struct AdaptiveSolverStrategy
    catalogue::Vector{AlgorithmRecommendation}
end

#==============================================================================#
# Solver Catalogue for Adaptive Capabilities
#==============================================================================#

function build_adaptive_solver_catalogue()
    AlgorithmRecommendation[
        AlgorithmRecommendation(QNDF(), 8.5, ADAPTIVE;
            description = "Quasi-Newton-based adaptive order solver (non-SL_STIFF or mildly SL_STIFF).",
            memory_efficiency = 0.7,
            stability_score = 0.8,
            stiffness_range = (SL_NON_STIFF, SL_STIFF),
            supports_events = true,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(CVODE_Adams(), 8.0, ADAPTIVE;
            description = "SUNDIALS CVODE with Adams-Moulton adaptive multistep integration.",
            memory_efficiency = 0.9,
            computational_cost = 0.6,
            stability_score = 0.75,
            stiffness_range = (SL_NON_STIFF, SL_MILDLY_STIFF),
            handles_mass_matrix = true,
            supports_events = true,
            references = ["https://computing.llnl.gov/projects/sundials"]),

        AlgorithmRecommendation(CVODE_BDF(), 9.0, ADAPTIVE;
            description = "SUNDIALS CVODE with BDF and adaptive order/stiffness control.",
            memory_efficiency = 0.8,
            computational_cost = 0.7,
            stability_score = 0.95,
            stiffness_range = (SL_MILDLY_STIFF, SL_EXTREMELY_STIFF),
            handles_mass_matrix = true,
            supports_events = true,
            references = ["https://computing.llnl.gov/projects/sundials"]),

        AlgorithmRecommendation(VCABM(), 7.5, ADAPTIVE;
            description = "Variable-coefficient Adams–Bashforth–Moulton method (explicit).",
            memory_efficiency = 0.85,
            computational_cost = 0.4,
            stability_score = 0.6,
            stiffness_range = (SL_NON_STIFF, SL_MILDLY_STIFF),
            supports_events = true,
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

    ]
end

#==============================================================================#
# Recommendation Function
#==============================================================================#

"""
    get_adaptive_recommendations(analysis::SystemAnalysis; rtol=1e-6,
                                 prefer_memory=false, prefer_stability=true)

Return a sorted list of adaptive-capable algorithms suitable for the given system.
"""
function get_adaptive_recommendations(analysis::SystemAnalysis;
                                      rtol::Float64=1e-6,
                                      prefer_memory::Bool=false,
                                      prefer_stability::Bool=true)

    catalogue = build_adaptive_solver_catalogue()

    filtered = filter(rec -> is_applicable(rec, analysis, rtol), catalogue)

    scored = sort(filtered; by = rec -> -compute_adjusted_priority(rec, analysis;
                                        prefer_memory=prefer_memory,
                                        prefer_stability=prefer_stability))

    return scored
end

export AdaptiveSolverStrategy, get_adaptive_recommendations

end # module


