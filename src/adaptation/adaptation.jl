# adaptation/adaptation.jl - Top-level adaptation framework

module Adaptation

using ..FCore: AbstractAdaptationStrategy, AdaptationState, SystemAnalysis, StepInfo, AlgorithmRecommendation, compute_adjusted_priority

# Include individual adaptation strategy modules
include("performance_adaptation.jl")
include("stability_adaptation.jl")
include("convergence_adaptation.jl")
include("memory_adaptation.jl")
include("parallel_adaptation.jl")
include("hybrid_adaptation.jl")

# Re-export all strategies
using .PerformanceAdaptation: PerformanceAdaptationStrategy, create_adaptive_composite
using .StabilityAdaptation: StabilityAdaptationStrategy
using .ConvergenceAdaptation: ConvergenceAdaptationStrategy
using .MemoryAdaptation: MemoryAdaptationStrategy
using .ParallelAdaptation: ParallelAdaptationStrategy
using .HybridAdaptation: HybridAdaptationStrategy

export PerformanceAdaptationStrategy, StabilityAdaptationStrategy, ConvergenceAdaptationStrategy,
       MemoryAdaptationStrategy, ParallelAdaptationStrategy, HybridAdaptationStrategy,
       AdaptationController, register_strategy!, adapt!, create_adaptive_composite

"""
    AdaptationController

Orchestrates multiple adaptation strategies and resolves conflicts using weighted voting.
"""
mutable struct AdaptationController
    strategies::Vector{Tuple{AbstractAdaptationStrategy, Float64}} # (strategy, weight)
    history::Vector{Any}
    
    AdaptationController() = new(Tuple{AbstractAdaptationStrategy, Float64}[], Any[])
end

"""
    register_strategy!(controller::AdaptationController, strategy::AbstractAdaptationStrategy, weight::Float64=1.0)
"""
function register_strategy!(controller::AdaptationController, strategy::AbstractAdaptationStrategy, weight::Float64=1.0)
    push!(controller.strategies, (strategy, weight))
end

"""
    adapt!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo) -> AlgorithmRecommendation
"""
function adapt!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo)
    votes = Dict{Any, Float64}() # algorithm_type => weighted_score
    recs = Dict{Any, AlgorithmRecommendation}()
    
    for (strategy, strategy_weight) in controller.strategies
        rec = adapt!(strategy, analysis, step)
        if rec !== nothing
            alg_type = typeof(rec.algorithm)
            priority = compute_adjusted_priority(rec, analysis)
            
            score = strategy_weight * priority
            votes[alg_type] = get(votes, alg_type, 0.0) + score
            recs[alg_type] = rec
        end
    end
    
    if isempty(votes)
        return nothing
    end
    
    best_alg_type = argmax(votes)
    return recs[best_alg_type]
end

# Base interface fallback
function adapt!(strategy::AbstractAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    return nothing
end

end # module Adaptation
