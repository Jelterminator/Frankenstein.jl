module Timescales

using LinearAlgebra
using ...FCore: SystemAnalysis, StepInfo
using ...Utilities.Jacobians: compute_jacobian

export compute_timescales, update_timescales!

function compute_timescales(prob, u=prob.u0, t=prob.tspan[1]; J=nothing)
    if J === nothing
        J = compute_jacobian(prob.f, u, prob.p, t)
    end
    # eigvals only works for dense matrices.
    evs = eigvals(Array(J))
    reals = real.(evs)
    timescales = 1.0 ./ max.(abs.(reals), 1e-10)
    return timescales
end

function update_timescales!(sa::SystemAnalysis, step_info::StepInfo; max_iterations=2)
    # Use power iteration to estimate dominant eigenvalue (inverse timescale)
    J = sa.jacobian
    if J === nothing
        return nothing  # Skip update if no Jacobian available
    end
    v = randn(size(J, 1))
    for _ in 1:max_iterations
        v = J * v
        v /= norm(v)
    end
    dominant_eig = norm(J * v) / norm(v)  # Rayleigh quotient
    dominant_timescale = 1.0 / max(abs(dominant_eig), 1e-10)
    
    # Scale existing timescales proportionally
    scale_factor = dominant_timescale / maximum(sa.timescales)
    sa.timescales .*= scale_factor
    return nothing
end

end # module Timescales
