module MemoryAdaptation
using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, StepInfo
export MemoryAdaptationStrategy, adapt!
struct MemoryAdaptationStrategy <: AbstractAdaptationStrategy end
function adapt!(strategy::MemoryAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo) return nothing end
end
