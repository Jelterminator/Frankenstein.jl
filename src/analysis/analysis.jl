# analysis.jl
"""
    analysis.jl
    
Main analysis module for system characterization.
"""

module Analysis

using SciMLBase
using LinearAlgebra
using SparseArrays
using ForwardDiff

using ..FCore
using ..Utilities.Jacobians

include("sparsity_analysis.jl")
using .Sparsity

include("stiffness_analysis.jl")
using .Stiffness

include("condition_analysis.jl")
using .Condition

include("convergence_analysis.jl")
using .Convergence

export analyze_system_structure, light_pulse, heavy_diagnostic!

"""
    analyze_system_structure(prob::ODEProblem) -> SystemAnalysis
"""
function analyze_system_structure(prob::ODEProblem)
    f = prob.f
    u0 = prob.u0
    p = prob.p
    t0 = prob.tspan[1]

    # 1. System size analysis
    system_size = length(u0)

    # 2. Sparsity detection
    is_sparse, density, proto = Sparsity.detect_sparsity_patterns(prob)
    
    # 3. Compute initial numerical Jacobian for further analysis
    J = try
        compute_jacobian(f, u0, p, t0)
    catch e
        @warn "Jacobian computation failed: $e. Using finite differences."
        finite_difference_jac(f, u0, p, t0, inplace=SciMLBase.isinplace(f))
    end
    
    # IMPROVED: Use the user's prototype if it exists, otherwise use what we derived
    sparsity_pattern = proto !== nothing ? proto : (is_sparse ? sparse(J) : nothing)
    
    # 4. Perform further analyses (Lazy if system is small/medium)
    stiffness = 0.0
    timescales = Float64[]
    coupling = 0.0
    condition = 1.0

    if system_size < 200
        stiffness = initial_stiffness_estimate(f, u0, p, J0=J)
        timescales = initial_timescale_analysis(J)
        coupling = estimate_coupling_strength(J)
        condition = try cond(collect(J)) catch; 1.0 end
    else
        # Large sparse systems: avoid expensive dense analysis, use opnorm proxy
        stiffness = initial_stiffness_estimate(f, u0, p, J0=J)
        timescales = Float64[]
        coupling = 0.0 # Skip for large systems
        condition = try opnorm(J, 1) catch; 1.0 end
        @info "[Frankenstein Analysis] Large system detected. Using sparse heuristics and opnorm proxy."
    end

    @info "[Frankenstein Analysis] System Size: $system_size | Sparse: $is_sparse | Density: $(round(density*100, digits=2))%"

    return SystemAnalysis{Float64}(
        stiffness, stiffness > 1e4, sparsity_pattern, timescales, coupling, condition, 
        system_size, is_sparse, J, 0, -100, t0, 0, 0.0, 0, 0, -100, 50, 10000, EXPLICIT, Dict{Symbol, Any}()
    )
end

function initial_timescale_analysis(J)
    if size(J, 1) > 500
        return Float64[] # Skip full eigen for large systems in step 0
    end
    try
        evals = eigvals(collect(J))
        return abs.(1.0 ./ (evals .+ 1e-12)) 
    catch
        return Float64[]
    end
end

function estimate_coupling_strength(J)
    norm_J = norm(J)
    if norm_J == 0 return 0.0 end
    
    J_off = copy(J)
    for i in 1:min(size(J, 1), size(J, 2))
        J_off[i, i] = 0.0
    end
    return norm(J_off) / norm_J
end

#==============================================================================#
# Tiered Diagnostic Pulse & Probe
#==============================================================================#

"""
    light_pulse(analysis::SystemAnalysis, step::StepInfo) -> Bool

The 'heartbeat' monitor. Returns true if something looks wrong enough to trigger
a heavy diagnosis. O(N) complexity.
"""
function light_pulse(analysis::SystemAnalysis, step::StepInfo)
    analysis.current_step = step.nsteps
    
    # 1. Hard Failures (Emergency Trigger)
    new_rejects = step.rejects - analysis.last_reject_count
    steps_since_update = analysis.current_step - analysis.last_update_step
    
    # Trigger if we have >= 5 rejections AND they constitute a high rejection rate (> 50%)
    # This prevents normal occasional rejections in stiff solvers from spamming diagnostics over long cooldowns.
    if new_rejects >= 5 && new_rejects > 0.5 * steps_since_update
        @info "[Pulse] Trigger: Emergency - High Rejection Rate ($new_rejects rejects in $steps_since_update steps)"
        return true
    end
    
    # 2. Adaptive Cooldown
    if (analysis.current_step - analysis.last_update_step) < analysis.diagnostic_cooldown
        # In cooldown, block all diagnostics. Emergencies (>=3 rejects) are already caught above.
        return false
    end
    
    # 3. RICE'S ASP: Hysteresis Border Check (The "New Model")
    # This is the cheap O(N) check that prevents "Diagnostic Spam"
    f = step.prob.f
    cheap_ρ = Stiffness.cheap_spectral_radius_estimate(f, step.u, step.p, step.t)
    
    # Update current estimate in analysis (without heavy diagnostic)
    analysis.stiffness_ratio = cheap_ρ 
    
    current_cat = analysis.current_category
    
    if current_cat == EXPLICIT || current_cat == STABILIZED_EXPLICIT
        # Crossing UP into Stiff territory?
        if cheap_ρ > BORDER_STIFF_UP
            @info "[Pulse] Trigger: Border Crossing UP ($cheap_ρ > $BORDER_STIFF_UP)"
            return true
        end
    elseif current_cat == STIFF || current_cat == SPARSE
        # Crossing DOWN into Explicit territory?
        if cheap_ρ < BORDER_STIFF_DOWN
            @info "[Pulse] Trigger: Border Crossing DOWN ($cheap_ρ < $BORDER_STIFF_DOWN)"
            return true
        end
    end
    
    # 4. Feature Space Jump (Massive derivative spike)
    norm_du = norm(step.du)
    if analysis.last_norm_du > 0 && norm_du > 1000.0 * analysis.last_norm_du
        @info "[Pulse] Trigger: Feature Space Jump (Norm Spike)"
        analysis.last_norm_du = norm_du 
        return true
    end
    analysis.last_norm_du = norm_du
    
    # 5. Long-term Watchdog (Amortized)
    if (analysis.current_step - analysis.last_update_step) >= analysis.watchdog_interval
        @debug "[Pulse] Trigger: Periodic Watchdog (interval: $(analysis.watchdog_interval))"
        return true
    end
    
    return false
end

"""
    heavy_diagnostic!(analysis::SystemAnalysis, step::StepInfo)

The 'Probe'. Executes expensive trait discovery (Jacobian, stiffness, etc.)
after being triggered by a pulse.
"""
function heavy_diagnostic!(analysis::SystemAnalysis, step::StepInfo)
    @info "[Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=$(step.t)"
    
    # 1. Update traits
    update_stiffness!(analysis, step)
    Condition.update_condition_number!(analysis, step)
    
    # 2. Update Coupling (only if system is small/medium)
    if analysis.system_size < 500
        analysis.coupling_strength = estimate_coupling_strength(analysis.jacobian)
    end
    
    analysis.last_update_step = analysis.current_step
    analysis.last_update_t = step.t
    analysis.last_reject_count = step.rejects
    @info "[Frankenstein] Results: Stiffness=$(round(analysis.stiffness_ratio, digits=2)) | Coupling=$(round(analysis.coupling_strength, digits=2))"
    
    return nothing
end

end # module
