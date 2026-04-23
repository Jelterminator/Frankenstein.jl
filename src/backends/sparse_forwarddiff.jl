# sparse_forwarddiff.jl
"""
    sparse_forwarddiff.jl
    
Sparse ForwardDiff backend implementation.
"""

using SparseDiffTools
using ForwardDiff
using SparseArrays
using ADTypes

"""
    configure_sparse_forwarddiff(sparsity_pattern=nothing; chunk_size=nothing)

Configure sparse ForwardDiff backend with optional sparsity pattern and chunk size.
"""
function configure_sparse_forwarddiff(sparsity_pattern=nothing; chunk_size=nothing)
    if chunk_size === nothing
        return AutoSparseForwardDiff(sparsity_pattern)
    else
        return AutoSparseForwardDiff(sparsity_pattern, Val(chunk_size))
    end
end

"""
    detect_sparsity_pattern(f, x)

Automatically detect sparsity pattern for a function.
"""
function detect_sparsity_pattern(f, x)
    try
        # Use SparseDiffTools to detect sparsity
        return jacobian_sparsity(f, x)
    catch e
        @warn "Could not detect sparsity pattern: $e"
        return nothing
    end
end

"""
    optimize_chunk_size(n, sparsity_pattern=nothing)

Determine optimal chunk size for ForwardDiff based on problem size and sparsity.
"""
function optimize_chunk_size(n, sparsity_pattern=nothing)
    if sparsity_pattern !== nothing
        # For sparse problems, use smaller chunks
        return min(12, max(1, n ÷ 10))
    else
        # For dense problems, use ForwardDiff's default heuristic
        return min(12, n)
    end
end
