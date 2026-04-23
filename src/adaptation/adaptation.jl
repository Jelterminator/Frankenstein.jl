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
    update_weights!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo)

Dynamically adjust strategy weights based on system state. 
- High rejects/stiffness -> Boost StabilityAdaptation
- Large dt/Success -> Boost PerformanceAdaptation
"""
function update_weights!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo)
    for i in 1:length(controller.strategies)
        strategy, base_weight = controller.strategies[i]
        
        multiplier = 1.0
        
        if strategy isa StabilityAdaptationStrategy
            # Boost stability if we are hitting stiffness or rejects
            if analysis.stiffness_ratio > 1e4 || step.rejects > 0
                multiplier = 2.5
            end
        elseif strategy isa PerformanceAdaptationStrategy
            # Boost performance if the solver is healthy and cruising
            if step.rejects == 0 && step.dt > 1e-4
                multiplier = 1.8
            end
        end
        
        # We don't overwrite base_weight, we just mutate the active weight if needed
        # Actually, let's keep the strategies as a vector of mutable objects or 
        # just calculate the active weight here.
    end
end

"""
    adapt!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo) -> AlgorithmRecommendation
"""
function adapt!(controller::AdaptationController, analysis::SystemAnalysis, step::StepInfo)
    votes = Dict{Any, Float64}() # algorithm_type => weighted_score
    recs = Dict{Any, AlgorithmRecommendation}()
    
    for (strategy, base_weight) in controller.strategies
        # Calculate dynamic weight
        active_weight = base_weight
        if strategy isa StabilityAdaptationStrategy && (analysis.stiffness_ratio > 1e4 || step.rejects > 0)
            active_weight *= 3.0 # Emergency stability boost
        elseif strategy isa PerformanceAdaptationStrategy && step.rejects == 0
            active_weight *= 1.5 # Cruising performance boost
        end

        rec = adapt!(strategy, analysis, step)
        if rec !== nothing
            alg_type = typeof(rec.algorithm)
            priority = compute_adjusted_priority(rec, analysis)
            
            score = active_weight * priority
            votes[alg_type] = get(votes, alg_type, 0.0) + score
            recs[alg_type] = rec
        end
    end
    
    if isempty(votes)
        return nothing
    end
    
    best_alg_type = argmax(votes)
    # Logging for transparency
    @debug "[Adaptation] Best candidate: $best_alg_type with score $(votes[best_alg_type])"
    
    return recs[best_alg_type]
end

# Base interface fallback
function adapt!(strategy::AbstractAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    return nothing
end

end # module Adaptation
