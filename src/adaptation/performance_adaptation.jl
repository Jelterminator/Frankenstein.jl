## adaptation/performance_adaptation.jl
module PerformanceAdaptation

using OrdinaryDiffEq
using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, AlgorithmRecommendation, StepInfo
using ...Monitoring: create_tiered_choice_function
using ...Solvers: get_all_recommendations, select_algorithm

export PerformanceAdaptationStrategy, adapt!, create_adaptive_composite

"""
    PerformanceAdaptationStrategy

Adaptation strategy that tunes solver backends and parameters for optimal runtime performance.
"""
struct PerformanceAdaptationStrategy <: AbstractAdaptationStrategy
end



import ..adapt!

function adapt!(strategy::PerformanceAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    # If the solver is cruising (no rejects, dt is large), look for performance gains
    if step.rejects == analysis.last_reject_count && step.dt > 1e-4
        # Stability-based transitions are now handled by the StabilityAdaptation Schmitt trigger
        # Performance adaptation can be extended here for things like internal tolerances or backend tweaks.
        return nothing
    end
    return nothing
end

"""
    create_adaptive_composite(analysis::SystemAnalysis)
"""
function create_adaptive_composite(analysis::SystemAnalysis)
    recs = get_all_recommendations(analysis)
    neighborhood = recs[1:min(3, length(recs))]
    algs = [r.algorithm for r in neighborhood]
    choice_function = create_tiered_choice_function(analysis)
    return CompositeAlgorithm(Tuple(algs), choice_function)
end

end # module
