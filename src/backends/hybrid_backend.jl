# hybrid_backend.jl
"""
    hybrid_backend.jl
    
Hybrid backend that can switch between different AD methods.
"""

using ADTypes

"""
    HybridBackend

A backend that can switch between different AD methods based on problem characteristics.
"""
mutable struct HybridBackend <: AbstractADType
    primary_backend::AbstractADType
    fallback_backend::AbstractADType
    switch_threshold::Float64
    current_backend::AbstractADType
end

"""
    configure_hybrid(primary, fallback; switch_threshold=1e-6)

Configure a hybrid backend with primary and fallback AD methods.
"""
function configure_hybrid(primary, fallback; switch_threshold=1e-6)
    return HybridBackend(primary, fallback, switch_threshold, primary)
end

"""
    switch_backend(hybrid::HybridBackend, condition::Bool)

Switch between primary and fallback backends based on condition.
"""
function switch_backend(hybrid::HybridBackend, condition::Bool)
    if condition
        hybrid.current_backend = hybrid.fallback_backend
    else
        hybrid.current_backend = hybrid.primary_backend
    end
    return hybrid.current_backend
end

"""
    jacobian(hybrid::HybridBackend, f, x)

Compute Jacobian using the current backend in the hybrid system.
"""
function jacobian(hybrid::HybridBackend, f, x)
    try
        return jacobian(hybrid.current_backend, f, x)
    catch e
        @warn "Primary backend failed, switching to fallback: $e"
        switch_backend(hybrid, true)
        return jacobian(hybrid.current_backend, f, x)
    end
end
