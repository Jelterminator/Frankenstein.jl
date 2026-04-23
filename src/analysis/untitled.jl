# coupling_analysis.jl

using LinearAlgebra

function compute_coupling_strength(prob, u=prob.u0, t=prob.tspan[1])
    f = prob.f
    p = prob.p
    J = compute_jacobian(f, u, p, t)
    n = size(J, 1)
    sum_diag = sum(abs(J[i,i]) for i in 1:n)
    sum_off_diag = sum(abs(J[i,j]) for i in 1:n for j in 1:n if i != j)
    coupling_strength = sum_off_diag / max(sum_diag, 1e-10)
    return coupling_strength
end
