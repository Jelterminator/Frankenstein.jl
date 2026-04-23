module StabilityAdaptation

using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, AlgorithmRecommendation, StepInfo
using ...Solvers: compute_adjusted_priority, select_algorithm

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
    
    # If the system has more than 10 rejects or a high stiffness ratio, trigger re-selection
    if local_stiffness > strategy.stiffness_threshold || step.rejects > 0
        @info "[Adaptation] Stability crisis detected! Re-evaluating algorithm candidates..."
        return select_algorithm(analysis; prefer_stability=true)
    end
    return nothing
end

end # module
