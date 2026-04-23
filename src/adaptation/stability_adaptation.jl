module StabilityAdaptation

using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, AlgorithmRecommendation, StepInfo
using ...Solvers: compute_adjusted_priority, select_best_algorithm

export StabilityAdaptationStrategy, adapt!

"""
    StabilityAdaptationStrategy
"""
struct StabilityAdaptationStrategy <: AbstractAdaptationStrategy
    stiffness_threshold::Float64
    StabilityAdaptationStrategy(threshold::Float64=1e4) = new(threshold)
end



function adapt!(strategy::StabilityAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    local_stiffness = analysis.stiffness_ratio
    if local_stiffness > strategy.stiffness_threshold || step.rejects > 2
        return select_best_algorithm(analysis)
    end
    return nothing
end

end # module
