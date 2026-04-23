# enzyme_backend.jl
"""
    enzyme_backend.jl
    
Enzyme.jl backend implementation for reverse-mode AD.
"""

using ADTypes

# Conditional loading of Enzyme
const ENZYME_AVAILABLE = try
    import Enzyme
    true
catch
    false
end

# Implement jacobian for Enzyme
function jacobian(::AutoEnzyme, f, x)
    if !ENZYME_AVAILABLE
        error("Enzyme not available")
    end
    # Enzyme normally wants to compute in-place or returns result
    # For simplicity in this interface:
    dx = zero(x)
    J = zeros(eltype(x), length(f(x)), length(x))
    # This is a placeholder, Enzyme usage is complex
    # Usually: Enzyme.jacobian(f, x)
    return Enzyme.jacobian(Enzyme.Forward, f, x)
end


"""
    configure_enzyme(mode=:reverse)

Configure Enzyme backend for automatic differentiation.
"""
function configure_enzyme(mode=:reverse)
    if !ENZYME_AVAILABLE
        @warn "Enzyme.jl not available, falling back to ForwardDiff"
        return AutoForwardDiff()
    end
    
    if mode == :reverse
        return AutoEnzyme()
    elseif mode == :forward
        return AutoEnzyme()  # Enzyme supports both modes
    else
        error("Unsupported Enzyme mode: $mode")
    end
end

"""
    is_enzyme_suitable(problem_size, memory_constraints=false)

Check if Enzyme is suitable for the given problem.
"""
function is_enzyme_suitable(problem_size, memory_constraints=false)
    if !ENZYME_AVAILABLE
        return false
    end
    
    # Enzyme is good for large problems where reverse mode is beneficial
    return problem_size >= 50 && !memory_constraints
end
