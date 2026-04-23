module Sparsity

using SciMLBase, ForwardDiff, SparseArrays, Symbolics
using Logging
using ...FCore: SystemAnalysis

"""
    detect_sparsity_patterns(prob::ODEProblem) -> Union{AbstractMatrix, Nothing}
Detects the sparsity pattern of the Jacobian.
"""
function detect_sparsity_patterns(prob::SciMLBase.ODEProblem)
    f = prob.f

    # Method 1: Use jac_prototype if provided
    # Note: we don't check has_jac(f) because jac_prototype can be provided 
    # even without an analytical Jacobian function.
    if hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing
        return f.jac_prototype isa SparseMatrixCSC ? copy(f.jac_prototype) : sparse(f.jac_prototype)
    end


    # Method 2: Symbolic detection
    if hasfield(typeof(f), :sys) && hasfield(typeof(f.sys), :eqs)
        try
            vars = f.sys.states
            return Symbolics.jacobian_sparsity(f.sys.eqs, vars)
        catch e
            @warn "Symbolic sparsity detection failed: $e."
        end
    end

    # Method 3: Numerical detection for small systems
    if length(prob.u0) < 1000
        try
            u0 = prob.u0
            p = prob.p
            t_sample = prob.tspan[1]
            if SciMLBase.isinplace(prob)
                ad_func = (u) -> begin
                    du = similar(u) # Create du of same type as u (e.g. Dual)
                    f(du, u, p, t_sample)
                    return du
                end
            else
                ad_func = (u) -> f(u, p, t_sample)
            end
            J_numerical = ForwardDiff.jacobian(ad_func, u0)
            return sparse(J_numerical .!= 0)
        catch e
            @warn "Numerical sparsity detection failed: $e"
        end
    end

    return nothing
end

end # module Sparsity
