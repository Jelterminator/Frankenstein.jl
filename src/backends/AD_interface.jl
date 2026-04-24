# AD_interface.jl
"""
    AD_interface.jl
    
Common interface for all automatic differentiation backends.
"""

using ADTypes
import ADTypes: jacobian_sparsity, AbstractADType, NoSparsityDetector, NoColoringAlgorithm
using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff
using SparseDiffTools

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

function jacobian(backend::AutoSparse, f, x)
    # Generic sparse Jacobian computation using SparseDiffTools
    # This is useful for benchmarking and small-scale testing
    n = length(x)
    
    # Get or detect sparsity
    sparsity = if hasproperty(backend, :sparsity_detector) && !(backend.sparsity_detector isa NoSparsityDetector)
        # Use provided pattern if it's already a matrix
        if backend.sparsity_detector isa AbstractMatrix
            backend.sparsity_detector
        else
            jacobian_sparsity(f, x, backend.sparsity_detector)
        end
    else
        # Fallback to structural detection
        try
            SparseDiffTools.jacobian_sparsity(f, x)
        catch
            # Last resort: numerical
            sparse(ForwardDiff.jacobian(f, x))
        end
    end
    
    # Compute coloring
    colorvec = if hasproperty(backend, :coloring_algorithm) && !(backend.coloring_algorithm isa NoColoringAlgorithm)
        SparseDiffTools.matrix_colors(sparsity, backend.coloring_algorithm)
    else
        SparseDiffTools.matrix_colors(sparsity)
    end
    
    # Build cache and compute
    jac_cache = ForwardDiffColorJacobianCache(f, x; colorvec=colorvec, sparsity=sparsity)
    J = copy(sparsity)
    SparseDiffTools.forwarddiff_color_jacobian!(J, f, x, jac_cache)
    return J
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
        ad_jac = jacobian(backend, f, x) # Fixed: use our jacobian
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
    if backend isa AutoSymbolics
        return problem_size <= 20
    end
    
    return false
end
