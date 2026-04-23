module Monitoring

using ..FCore
export NullMonitor

include("adaptation.jl")
using .AdaptationMonitoring: TieredAdaptationState, create_tiered_choice_function
export TieredAdaptationState, create_tiered_choice_function

struct NullMonitor <: AbstractPerformanceMonitor end

end # module Monitoring
