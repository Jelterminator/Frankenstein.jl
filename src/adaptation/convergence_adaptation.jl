module ConvergenceAdaptation
using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, StepInfo
export ConvergenceAdaptationStrategy, adapt!
struct ConvergenceAdaptationStrategy <: AbstractAdaptationStrategy
    error_tolerance::Float64
end
function adapt!(strategy::ConvergenceAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    return nothing
end
end
