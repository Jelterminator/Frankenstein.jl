module StabilityAdaptation

using ...FCore: SystemAnalysis, AbstractAdaptationStrategy, AlgorithmRecommendation, StepInfo, SolverCategory, EXPLICIT, STABILIZED_EXPLICIT, STIFF, SPARSE, BORDER_STIFF_UP, BORDER_STIFF_DOWN
using ...Solvers: compute_adjusted_priority, select_algorithm

export StabilityAdaptationStrategy, adapt!

"""
    StabilityAdaptationStrategy
"""
struct StabilityAdaptationStrategy <: AbstractAdaptationStrategy
    stiffness_threshold::Float64
    StabilityAdaptationStrategy(threshold::Float64=2000.0) = new(threshold)
end

import ..adapt!
function adapt!(strategy::StabilityAdaptationStrategy, analysis::SystemAnalysis, step::StepInfo)
    local_stiffness = analysis.stiffness_ratio
    current_cat = analysis.current_category
    
    # RICE'S ASP: Hysteresis (Schmitt Trigger) logic
    # Transition thresholds:
    # EXPLICIT -> STIFF: stiffness > BORDER_STIFF_UP
    # STIFF -> EXPLICIT: stiffness < BORDER_STIFF_DOWN
    
    should_switch = false
    reason = ""
    
    if current_cat == EXPLICIT || current_cat == STABILIZED_EXPLICIT
        if local_stiffness > BORDER_STIFF_UP || step.rejects > analysis.last_reject_count
            should_switch = true
            reason = local_stiffness > BORDER_STIFF_UP ? "Stiffness upper threshold crossed ($local_stiffness > $BORDER_STIFF_UP)" : "Step rejection in explicit mode"
        end
    elseif current_cat == STIFF || current_cat == SPARSE
        if local_stiffness < BORDER_STIFF_DOWN && step.rejects == 0
            should_switch = true
            reason = "Stiffness lower threshold crossed ($local_stiffness < $BORDER_STIFF_DOWN)"
        end
    else
        # Default fallback for other categories (COMPOSITE, etc.)
        if local_stiffness > strategy.stiffness_threshold || step.rejects > analysis.last_reject_count
            should_switch = true
            reason = "General stability criteria"
        end
    end
    
    if should_switch
        @info "🔄 [Adaptation] Hysteresis Trigger: $reason. Re-evaluating algorithm..."
        return select_algorithm(analysis; prefer_stability=true)
    end
    
    return nothing
end

end # module
