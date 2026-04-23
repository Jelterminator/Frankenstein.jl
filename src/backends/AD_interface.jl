# AD_interface.jl
"""
    AD_interface.jl
    
Common interface for all automatic differentiation backends.
"""

using ADTypes
using LinearAlgebra
using SparseArrays

"""
    jacobian(backend, f, x)

Generic function for computing Jacobians using different AD backends.
Backends should extend this function.
"""
function jacobian end

# Default implementations for standard backends
function jacobian(::AutoForwardDiff, f, x)
    return ForwardDiff.jacobian(f, x)
end

function jacobian(::AutoFiniteDiff, f, x)
    return FiniteDiff.finite_difference_jacobian(f, x)
end


"""
    evaluate_backend_performance(backend, f, x, sparsity_pattern=nothing)

Benchmark an AD backend's performance on a given function and input.
Returns BackendPerformanceMetrics.
"""
function evaluate_backend_performance(backend::AbstractADType, f, x, sparsity_pattern=nothing)
    T = eltype(x)
    metrics = BackendPerformanceMetrics{T}()
    
    try
        # Time Jacobian computation
        jacobian_time = @elapsed begin
            jac = jacobian(backend, f, x)
        end
        metrics.jacobian_time = jacobian_time
        
        # Estimate memory allocation (simplified)
        n = length(x)
        if sparsity_pattern !== nothing
            metrics.memory_allocation = nnz(sparsity_pattern) * sizeof(T)
        else
            metrics.memory_allocation = n^2 * sizeof(T)
        end
        
        # Compute accuracy metric (compare with finite differences)
        fd_jac = FiniteDiff.finite_difference_jacobian(f, x)
        ad_jac = ADTypes.jacobian(backend, f, x)
        metrics.accuracy = 1.0 / (1.0 + norm(fd_jac - ad_jac))
        
        # Sparsity efficiency
        if sparsity_pattern !== nothing
            metrics.sparsity_efficiency = nnz(sparsity_pattern) / length(sparsity_pattern)
        else
            metrics.sparsity_efficiency = 1.0
        end
        
        metrics.stability = 1.0  # Placeholder for stability metric
        
    catch e
        @warn "Backend evaluation failed: $e"
        metrics.jacobian_time = T(Inf)
    end
    
    return metrics
end

"""
    is_backend_suitable(backend, problem_size, sparsity_ratio)

Check if a backend is suitable for the given problem characteristics.
"""
function is_backend_suitable(backend::AbstractADType, problem_size::Int, sparsity_ratio::Float64)
    # ForwardDiff is efficient for small to medium problems
    if backend isa AutoForwardDiff
        return problem_size <= 100
    end
    
    # Finite differences are universal but slow
    if backend isa AutoFiniteDiff
        return true
    end
    
    # Sparse ForwardDiff for sparse problems
    if backend isa AutoSparseForwardDiff
        return sparsity_ratio < 0.1  # Less than 10% dense
    end
    
    # Enzyme for large problems (when available)
    if backend isa AutoEnzyme
        return problem_size >= 50
    end
    
    # Symbolic for small analytical problems
    if backend isa AutoSymbolic
        return problem_size <= 20
    end
    
    return false
end
