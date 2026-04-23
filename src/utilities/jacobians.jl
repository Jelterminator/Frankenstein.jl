module Jacobians

using SciMLBase, ForwardDiff

export finite_difference_jac, compute_jacobian

"""
    finite_difference_jac(f, u, p, t) -> J
Compute a (dense) Jacobian via central finite differences.
"""
function finite_difference_jac(f, u, p, t; inplace=false)
    n = length(u)
    T = eltype(u)
    J = zeros(T, n, n)
    δ = sqrt(eps(T))
    
    if inplace
        f_plus = similar(u)
        f_minus = similar(u)
        for i in 1:n
            ui = u[i]
            u[i] = ui + δ
            f(f_plus, u, p, t)
            u[i] = ui - δ
            f(f_minus, u, p, t)
            u[i] = ui # Restore
            J[:, i] = (f_plus .- f_minus) ./ (2δ)
        end
    else
        for i in 1:n
            ui = u[i]
            u_plus = copy(u); u_plus[i] += δ
            u_minus = copy(u); u_minus[i] -= δ
            J[:, i] = (f(u_plus, p, t) .- f(u_minus, p, t)) ./ (2δ)
        end
    end
    return J
end

"""
    compute_jacobian(f, u, p, t) -> Matrix

Compute the Jacobian matrix. Uses ForwardDiff.jl with correct dual number handling.
"""
function compute_jacobian(f, u, p, t)
    if SciMLBase.has_jac(f)
        if f.jac !== nothing
            J = similar(u, length(u), length(u))
            if SciMLBase.isinplace(f)
                f.jac(J, u, p, t)
                return J
            else
                return f.jac(u, p, t)
            end
        end
    end

    try
        if SciMLBase.isinplace(f)
            ad_func! = (out, x) -> f(out, x, p, t)
            return ForwardDiff.jacobian(ad_func!, similar(u), u)
        else
            ad_func = x -> f(x, p, t)
            return ForwardDiff.jacobian(ad_func, u)
        end
    catch e
        @warn "Automatic differentiation failed: $e. Using finite differences."
        return finite_difference_jac(f, u, p, t, inplace=SciMLBase.isinplace(f))
    end
end

end # module Jacobians
