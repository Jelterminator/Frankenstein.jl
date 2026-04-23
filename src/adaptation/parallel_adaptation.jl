module ParallelAdaptation
using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, StepInfo
export ParallelAdaptationStrategy, adapt!
struct ParallelAdaptationStrategy <: AbstractAdaptationStrategy end
function adapt!(strategy::ParallelAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo) return nothing end
end
