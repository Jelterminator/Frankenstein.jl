module Coupling

using LinearAlgebra
using ...FCore: SystemAnalysis, StepInfo
using ...Utilities.Jacobians: compute_jacobian

export compute_coupling_strength, update_coupling_strength!

function compute_coupling_strength(prob, u=prob.u0, t=prob.tspan[1]; J=nothing)
    if J === nothing
        J = compute_jacobian(prob.f, u, prob.p, t)
    end
    # Simple measure: ratio of off-diagonal norm to diagonal norm
    J_mat = Array(J)
    diag_J = diag(J_mat)
    off_diag_J = J_mat - Diagonal(diag_J)
    return norm(off_diag_J) / max(norm(diag_J), 1e-10)
end

function update_coupling_strength!(sa::SystemAnalysis, step_info::StepInfo)
    # Heuristic update based on du norm changes
    sa.coupling_strength *= (1.0 + 0.1 * randn()) # Random walk for now
    return nothing
end

end # module Coupling
