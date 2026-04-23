module Sparsity

using SciMLBase, ForwardDiff, SparseArrays, Symbolics, SparseDiffTools
using Logging
using ...FCore: SystemAnalysis

"""
    detect_sparsity_patterns(prob::ODEProblem) -> (Bool, Float64, Union{AbstractMatrix, Nothing})
Detects the sparsity pattern and calculates density metrics.
"""
function detect_sparsity_patterns(prob::SciMLBase.ODEProblem)
    f = prob.f
    n = length(prob.u0)

    # Method 1: Robust property probe for jac_prototype or sparsity
    proto = nothing
    try
        if hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing
            proto = f.jac_prototype
        elseif hasproperty(f, :sparsity) && f.sparsity !== nothing
            proto = f.sparsity
        end
    catch
        # Fallback if property access is protected
    end

    if proto !== nothing
        pattern = proto isa SparseMatrixCSC ? copy(proto) : sparse(proto)
        density = nnz(pattern) / (n^2)
        return (density < 0.35, density, pattern)
    end


    # Method 2: Symbolic detection
    if hasfield(typeof(f), :sys) && hasfield(typeof(f.sys), :eqs)
        try
            vars = f.sys.states
            pattern = Symbolics.jacobian_sparsity(f.sys.eqs, vars)
            density = nnz(pattern) / (n^2)
            return (density < 0.35, density, pattern)
        catch e
            @warn "Symbolic sparsity detection failed: $e."
        end
    end

    # Method 3: Structural and Numerical detection for small systems
    if n < 1000
        try
            u0 = prob.u0
            p = prob.p
            t_sample = prob.tspan[1]
            
            if SciMLBase.isinplace(prob)
                ad_func = (u) -> begin
                    du = similar(u)
                    f(du, u, p, t_sample)
                    return du
                end
            else
                ad_func = (u) -> f(u, p, t_sample)
            end
            
            # Prioritize structural detection using SparseDiffTools (Recommended)
            pattern = try
                SparseDiffTools.jacobian_sparsity(ad_func, u0)
            catch
                # FIX: Perturb the state slightly to avoid accidental zeros at the initial condition
                u_rand = u0 .+ rand(eltype(u0), length(u0)) .* 1e-5
                J_numerical = ForwardDiff.jacobian(ad_func, u_rand)
                
                # Use a tiny threshold rather than absolute zero to account for floating point noise
                sparse(abs.(J_numerical) .> 1e-12)
            end
            
            density = nnz(pattern) / (n^2)
            return (density < 0.35, density, pattern)
        catch e
            @warn "Sparsity detection failed in Method 3: $e"
        end
    end

    return (false, 1.0, nothing)
end

end # module Sparsity
