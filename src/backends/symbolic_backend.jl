# symbolic_backend.jl
"""
    symbolic_backend.jl
    
Symbolic differentiation backend using Symbolics.jl.
"""

using Symbolics
using ADTypes
using SciMLBase


"""
    configure_symbolic()

Configure symbolic differentiation backend.
"""
function configure_symbolic()
    return AutoSymbolics()
end

"""
    jacobian(::AutoSymbolic, f, x)

Compute Jacobian using symbolic differentiation.
"""
function jacobian(::AutoSymbolics, f, x)
    # 0. Robust unwrapping for SciML functions to bypass FunctionWrappers
    # Symbolics cannot trace through a FunctionWrapper limited to Float64.
    raw_f = if f isa SciMLBase.ODEFunction
        f.f
    else
        f
    end

    # 1. Create symbolic variables
    n = length(x)
    vars = Symbolics.@variables $(Symbol.("x", 1:n))[1:n]
    
    # 2. Handle in-place vs out-of-place for tracing
    symbolic_f = if SciMLBase.isinplace(f)
        du = similar(vars, Num)
        raw_f(du, vars, nothing, 0.0) # Assume p=nothing, t=0.0 for tracing
        du
    else
        raw_f(vars, nothing, 0.0)
    end
    
    # Compute symbolic Jacobian
    J_symbolic = Symbolics.jacobian(symbolic_f, vars)
    
    # Substitute numerical values
    substitutions = Dict(vars[i] => x[i] for i in 1:n)
    J_numerical = Symbolics.substitute(J_symbolic, substitutions)
    
    # Convert to numerical array
    return Float64.(J_numerical)
end

"""
    is_symbolic_suitable(problem_size, is_analytical=false)

Check if symbolic differentiation is suitable for the problem.
"""
function is_symbolic_suitable(problem_size, is_analytical=false)
    # Symbolic is good for small analytical problems
    return problem_size <= 20 && is_analytical
end
