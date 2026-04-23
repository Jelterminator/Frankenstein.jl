module HybridAdaptation
using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, StepInfo
export HybridAdaptationStrategy, adapt!
struct HybridAdaptationStrategy <: AbstractAdaptationStrategy end
function adapt!(strategy::HybridAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo) return nothing end
end
