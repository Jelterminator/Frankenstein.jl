# specialty_solvers.jl - Specialized and niche solver strategies

module SpecialtySolvers

using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM, SPECIALTY
using OrdinaryDiffEq
using Sundials

#==============================================================================#
# Specialty Solver Strategy Definition
#==============================================================================#

"""
    SpecialtySolverStrategy

Encapsulates logic for recommending solvers with niche features like high-precision,
low-memory, or structure-preserving behavior.
"""
struct SpecialtySolverStrategy
    catalogue::Vector{AlgorithmRecommendation}
end

#==============================================================================#
# Solver Catalogue for Specialty Use Cases
#==============================================================================#

function build_specialty_solver_catalogue()
    AlgorithmRecommendation[

        AlgorithmRecommendation(Tsit5(), 7.5, SPECIALTY;
            description = "Tsitouras 5/4 Runge-Kutta method with low memory use, good for short accurate runs.",
            memory_efficiency = 0.95,
            computational_cost = 0.4,
            stability_score = 0.6,
            handles_sparse = false,
            supports_events = true,
            stiffness_range = (SL_NON_STIFF, SL_MILDLY_STIFF),
            references = ["https://doi.org/10.1007/s10543-011-0346-3"]),

        AlgorithmRecommendation(Rodas5(autodiff=false), 8.8, SPECIALTY;
            description = "Rodas5 with manual Jacobian mode. Stable and accurate for SL_STIFF + event-rich problems.",
            memory_efficiency = 0.7,
            computational_cost = 0.6,
            stability_score = 0.95,
            handles_sparse = true,
            supports_events = true,
            stiffness_range = (SL_STIFF, SL_EXTREMELY_STIFF),
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(SymplecticEuler(), 6.5, SPECIALTY;
            description = "Symplectic Euler integrator. Suitable for Hamiltonian/constrained mechanical systems.",
            memory_efficiency = 0.9,
            stability_score = 0.85,
            supports_events = false,
            stiffness_range = (SL_NON_STIFF, SL_MILDLY_STIFF),
            references = ["https://en.wikipedia.org/wiki/Symplectic_integrator"]),

        AlgorithmRecommendation(Vern9(), 9.2, SPECIALTY;
            description = "High-accuracy, high-order explicit Runge-Kutta method. Useful for precision-sensitive ODEs.",
            memory_efficiency = 0.6,
            stability_score = 0.75,
            computational_cost = 0.9,
            supports_events = true,
            stiffness_range = (SL_NON_STIFF, SL_MILDLY_STIFF),
            references = ["https://github.com/SciML/OrdinaryDiffEq.jl"]),

        AlgorithmRecommendation(CVODE_BDF(linear_solver=:GMRES), 8.0, SPECIALTY;
            description = "CVODE_BDF with GMRES for matrix-free SL_STIFF systems with event handling.",
            memory_efficiency = 0.7,
            computational_cost = 0.7,
            stability_score = 0.95,
            handles_sparse = true,
            handles_mass_matrix = true,
            supports_events = true,
            stiffness_range = (SL_STIFF, SL_EXTREMELY_STIFF),
            references = ["https://computing.llnl.gov/projects/sundials"])
    ]
end

#==============================================================================#
# Recommendation Function
#==============================================================================#

"""
    get_specialty_recommendations(analysis::SystemAnalysis; rtol=1e-6,
                                  prefer_memory=false, prefer_stability=true)

Return a sorted list of specialty solvers tailored to unique system characteristics.
"""
function get_specialty_recommendations(analysis::SystemAnalysis;
                                       rtol::Float64=1e-6,
                                       prefer_memory::Bool=false,
                                       prefer_stability::Bool=true)

    catalogue = build_specialty_solver_catalogue()

    filtered = filter(rec -> is_applicable(rec, analysis, rtol), catalogue)

    scored = sort(filtered; by = rec -> -compute_adjusted_priority(rec, analysis;
                                        prefer_memory=prefer_memory,
                                        prefer_stability=prefer_stability))

    return scored
end

export SpecialtySolverStrategy, get_specialty_recommendations

end # module


