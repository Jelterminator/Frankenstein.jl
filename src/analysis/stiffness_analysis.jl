module Stiffness

using LinearAlgebra, ForwardDiff, SciMLBase
using ...FCore: SystemAnalysis, StepInfo
using ...Utilities.Jacobians: compute_jacobian

export initial_stiffness_estimate, update_stiffness!

"""
    initial_stiffness_estimate(f, u0, p; J0=nothing) -> Float64
Estimate the stiffness ratio ||J|| * dt_max / 2.
"""
function initial_stiffness_estimate(f, u0, p; J0=nothing)
    if J0 === nothing
        J0 = compute_jacobian(f, u0, p, 0.0)
    end
    
    n = size(J0, 1)
    if n > 200
        # Power iteration for spectral radius (stiffness)
        v = rand(eltype(u0), n)
        ρ = 0.0
        try
            for _ in 1:10
                v = J0 * v
                v_norm = norm(v)
                if v_norm == 0 || isnan(v_norm)
                    break
                end
                v /= v_norm
            end
            ρ = norm(J0 * v)
        catch
            ρ = 1e6 # Fallback to stiff
        end
        return isnan(ρ) ? 1e6 : ρ
    else
        # Spectral radius via eigenvalues
        try
            λ = eigvals(Array(J0))
            stiffness = maximum(abs.(λ))
            return isnan(stiffness) ? 1e6 : stiffness
        catch
            return 1e6 # Fallback if eigvals fails
        end
    end
end


"""
    update_stiffness!(analysis::SystemAnalysis, step_info::StepInfo)
Update the stiffness ratio for the current step.
"""
function update_stiffness!(analysis::SystemAnalysis, step_info::StepInfo)
    u = step_info.u
    p = step_info.p
    t = step_info.t
    prob = step_info.prob
    f = prob.f
    
    # Update Jacobian if it's been too long or we have many rejects
    if analysis.current_step - analysis.last_jacobian_update > 20 || step_info.rejects > 2
        analysis.jacobian = compute_jacobian(f, u, p, t)
        analysis.last_jacobian_update = analysis.current_step
    end
    
    J = analysis.jacobian
    if J !== nothing
        # Power iteration to find dominant eigenvalue (spectral radius)
        n = size(J, 1)
        v = rand(eltype(u), n)
        for _ in 1:3
            v = J * v
            v /= norm(v)
        end
        ρ = norm(J * v)
        analysis.stiffness_ratio = ρ
        analysis.is_stiff = ρ > 1e4 # Threshold
    end
    return nothing
end

end # module Stiffness
