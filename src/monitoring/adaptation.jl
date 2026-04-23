# monitoring/adaptation.jl
module AdaptationMonitoring

using SciMLBase
using ...FCore: SystemAnalysis

export TieredAdaptationState, create_tiered_choice_function

"""
    TieredAdaptationState

Mutable state tracking for the tiered adaptation choice function.
Designed for zero-allocation performance in the hot loop.
"""
mutable struct TieredAdaptationState
    consecutive_rejections::Int
    steps_since_heavy_diag::Int
    current_alg_index::Int
    dt_history::Float64
end

function TieredAdaptationState()
    return TieredAdaptationState(0, 0, 1, 0.0)
end

"""
    estimate_local_stiffness(integrator)
"""
function estimate_local_stiffness(integrator)
    return 1.0 # Default non-stiff
end

"""
    create_tiered_choice_function(analysis::SystemAnalysis)

Returns a closure compatible with `OrdinaryDiffEq.CompositeAlgorithm`.
"""
function create_tiered_choice_function(analysis::SystemAnalysis)
    state = TieredAdaptationState()
    
    # The actual choice function passed to CompositeAlgorithm
    function choice_function(integrator)
        state.steps_since_heavy_diag += 1
        state.dt_history = integrator.dt
        
        # --- Tier 1: Mild Diagnostics (Every Step, O(1)) ---
        
        # 1. Step Rejection Tracking
        if !integrator.accept_step
            state.consecutive_rejections += 1
        else
            state.consecutive_rejections = 0
        end
        
        # 2. Newton Solver Load (if implicit)
        newton_load_flag = false
        if state.current_alg_index >= 2
            if hasattr(integrator.destats, :nnonliniter) && integrator.destats.nnonliniter > 7
                newton_load_flag = true
            end
        end
        
        # 3. Step Size Crashing
        if integrator.dt < 1e-6 && state.current_alg_index == 1
            state.current_alg_index = 2
            return state.current_alg_index
        end
        
        # 4. Failure Trigger
        failure_trigger = (state.consecutive_rejections >= 2) || newton_load_flag
        
        # --- Tier 2: Heavy Diagnostics (Triggered or Periodic) ---
        if failure_trigger || state.steps_since_heavy_diag > 1000
            state.steps_since_heavy_diag = 0
            
            est = estimate_local_stiffness(integrator)
            
            if est > 10000
                state.current_alg_index = 3
            elseif est > 100
                state.current_alg_index = 2
            else
                state.current_alg_index = 1
            end
            
            if failure_trigger && state.current_alg_index < 3
                state.current_alg_index += 1
            end
        end
        
        return state.current_alg_index
    end
    
    return choice_function
end

# Helper to check for fields in destats
hasattr(obj, sym) = hasfield(typeof(obj), sym)

end # module AdaptationMonitoring
