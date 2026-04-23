module Analysis

using ..FCore: SystemAnalysis, StepInfo
using ..Utilities.Jacobians
using SciMLBase
using SparseArrays, Symbolics, ForwardDiff, LinearAlgebra

# Include submodule files
include("sparsity_analysis.jl")
using .Sparsity: detect_sparsity_patterns

include("stiffness_analysis.jl")
using .Stiffness: initial_stiffness_estimate, update_stiffness!

include("timescale_analysis.jl")
using .Timescales: compute_timescales, update_timescales!

include("coupling_analysis.jl")
using .Coupling: compute_coupling_strength, update_coupling_strength!

include("condition_analysis.jl")
using .Condition: compute_condition_number, update_condition_number!

# Export functions
export analyze_system_structure, detect_sparsity_patterns
export initial_stiffness_estimate, update_stiffness!
export compute_timescales, update_timescales!
export compute_coupling_strength, update_coupling_strength!
export compute_condition_number, update_condition_number!
export needs_analysis_update!

"""
    analyze_system_structure(prob::SciMLBase.ODEProblem) -> SystemAnalysis

Performs an initial comprehensive analysis of the ODE problem.
"""
function analyze_system_structure(prob)
    u0 = prob.u0
    t0 = prob.tspan[1]
    p = prob.p
    f = prob.f

    # Compute system size and sparsity
    system_size = length(u0)
    sparsity = detect_sparsity_patterns(prob)
    is_sparse = sparsity !== nothing && nnz(sparsity) / (system_size^2) < 0.1

    # Compute Jacobian with fallback
    try
        J = compute_jacobian(f, u0, p, t0)
    catch e
        @warn "Jacobian computation failed: $e. Using finite differences."
        J = finite_difference_jac(f, u0, p, t0, inplace=SciMLBase.isinplace(f))
    end
    J = is_sparse ? sparse(J) : J

    # Perform analyses
    stiffness = initial_stiffness_estimate(f, u0, p, J0=J)
    timescales = compute_timescales(prob, u0, t0, J=J)
    coupling = compute_coupling_strength(prob, u0, t0, J=J)
    condition = compute_condition_number(prob, u0, t0, J=J)

    return SystemAnalysis{Float64}(
        stiffness, stiffness > 1e4, sparsity, timescales, coupling, condition, 
        system_size, is_sparse, J, 0, 0, 0, 0.0, 0, Any[]
    )
end

"""
    needs_analysis_update!(sa::SystemAnalysis, step_info::StepInfo; ...)

Determines which analysis variables need updating based on dynamic indicators.
"""
function needs_analysis_update!(sa::SystemAnalysis, step_info::StepInfo; 
        step_change_thresh=2.0, reject_thresh=0.2, norm_change_thresh=0.5, 
        stable_steps=10, min_update_interval=5)
    
    update_stiffness = false
    update_timescales = false
    update_coupling = false
    update_condition = false

    sa.current_step += 1
    current_step = sa.current_step
    last_update_step = sa.last_update_step

    # Skip updates if within minimum interval, unless forced by significant changes
    if current_step - last_update_step < min_update_interval && sa.stable_count < stable_steps
        return (stiffness=false, timescales=false, coupling=false, condition=false)
    end

    # Compute lightweight indicators
    dt_ratio = step_info.dt_prev > 0 ? step_info.dt_prev / step_info.dt : 1.0
    reject_rate = step_info.rejects / 10.0 # Heuristic
    norm_du = norm(step_info.du)
    
    norm_change = sa.last_norm_du > 0 ? abs(norm_du / sa.last_norm_du - 1) : 0.0
    sa.last_norm_du = norm_du

    significant_change = dt_ratio > step_change_thresh || reject_rate > reject_thresh || norm_change > norm_change_thresh

    if significant_change || sa.stable_count >= stable_steps
        update_stiffness = dt_ratio > step_change_thresh || reject_rate > reject_thresh
        update_timescales = norm_change > norm_change_thresh || dt_ratio > step_change_thresh
        update_coupling = norm_change > norm_change_thresh || reject_rate > reject_thresh
        update_condition = dt_ratio > step_change_thresh || reject_rate > reject_thresh

        if significant_change
            sa.stable_count = 0
        end

        if update_stiffness || update_timescales || update_coupling || update_condition
            sa.last_update_step = current_step
        end
    end

    return (stiffness=update_stiffness, timescales=update_timescales, 
            coupling=update_coupling, condition=update_condition)
end

end # module Analysis
