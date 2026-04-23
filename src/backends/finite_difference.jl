# finite_difference.jl
"""
    finite_difference.jl
    
Finite difference backend implementation.
"""

using FiniteDiff
using ADTypes

"""
    configure_finite_diff(; stepsize=nothing, fdtype=:central)

Configure finite difference backend with custom step size and type.
"""
function configure_finite_diff(; stepsize=nothing, fdtype=:central)
    if stepsize === nothing
        return AutoFiniteDiff(fdtype)
    else
        return AutoFiniteDiff(fdtype, Val(stepsize))
    end
end

"""
    adaptive_stepsize(x, f)

Compute adaptive step size for finite differences based on function scale.
"""
function adaptive_stepsize(x, f)
    # Simple heuristic: use relative step size based on function values
    fx = f(x)
    scale = maximum(abs, fx)
    return scale > 0 ? sqrt(eps()) * max(1, scale) : sqrt(eps())
end
