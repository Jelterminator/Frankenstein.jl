# backend_selector.jl
"""
    backend_selector.jl
    
Intelligent backend selection using multiple dispatch and theoretical cost models.
"""

using ADTypes
using LinearAlgebra
using SparseArrays
using LinearSolve
using SparseMatrixColorings

"""
    BackendSelection
"""
struct BackendSelection{AD, LS}
    ad_backend::AD
    linear_solver::LS
    performance_score::Float64
    selection_rationale::String
end

"""
    choose_backend(...)
"""
function choose_backend(analysis, 
                        available_backends;
                        disabled_backends::Dict{String, Int}=Dict{String, Int}())
    
    current_step = analysis.current_step
    
    # Filter out backends that are currently in their 1000-step cooling-off period
    filtered_backends = filter(available_backends) do b
        b_name = string(typeof(b))
        disabled_until = get(disabled_backends, b_name, -1)
        return current_step >= disabled_until
    end

    best_backend  = nothing
    best_score    = -Inf

    n       = analysis.system_size
    is_sp   = analysis.is_sparse
    
    sparsity = 1.0 
    if is_sp && analysis.sparsity_pattern !== nothing
        try
            sparsity = nnz(analysis.sparsity_pattern) / (n^2)
        catch
            sparsity = 0.1
        end
    end
    
    stiff = analysis.is_stiff || (!isnan(analysis.stiffness_ratio) && analysis.stiffness_ratio > 1000.0)

    for backend in filtered_backends
        score = evaluate_backend_score(backend, n, sparsity, stiff, is_sp)
        if score > best_score
            best_score     = score
            best_backend   = backend
        end
    end

    if best_backend === nothing
        # EMERGENCY FALLBACK: If everything is disabled, we return FiniteDiff
        # unless filtered_backends was completely empty (unexpected)
        best_backend = AutoFiniteDiff()
        best_score = -999.0
    end

    best_rationale = generate_rationale(best_backend, n, sparsity, stiff)
    lin_solver = select_linear_solver_for_backend(best_backend, n, sparsity, stiff)
    
    return BackendSelection(best_backend, lin_solver, best_score, best_rationale)
end

#==============================================================================#
# Multiple Dispatch Cost Models
#==============================================================================#

# Generic fallback
evaluate_backend_score(backend, n, sparsity, stiff, is_sp) = 1.0

function evaluate_backend_score(backend::AutoForwardDiff, n, sparsity, stiff, is_sp)
    # ForwardDiff complexity is O(N) dual numbers, or O(N^2) total cost for dense J
    # We penalize as N grows large.
    base_score = 100.0
    scaling_penalty = (n / 100.0)^1.5 # Grows significantly after n=100
    return base_score - scaling_penalty
end

function evaluate_backend_score(backend::AutoEnzyme, n, sparsity, stiff, is_sp)
    # Enzyme (Reverse Mode) has constant factor overhead but scales O(1) in dual-like passes
    # It wins for large dense systems.
    base_score = 50.0
    scale_bonus = (n / 300.0) * 20.0 # Becomes attractive as n increases
    return base_score + scale_bonus
end

# Internal helper to handle the wrapped ADTypes.AutoSparse
function evaluate_backend_score(backend::AutoSparse, n, sparsity, stiff, is_sp)
    # Sparse AD is the king of PDEs, but for small systems (n < 200), 
    # dense AD is often more robust and just as fast.
    if !is_sp
        return -500.0 # Heavy penalty for using sparse on dense
    end
    
    if n < 200
        # Reduce priority for small sparse systems to favor robust dense AD
        return 50.0
    end
    
    # Base score is extremely high for larger sparse systems
    base_score = 500.0
    
    # We add a bonus if the coloring algorithm is high-quality (like GreedyColoring)
    coloring_bonus = 0.0
    if hasproperty(backend, :coloring_algorithm) && backend.coloring_algorithm !== nothing
        coloring_bonus = 50.0
    end
    
    return base_score + coloring_bonus
end

function evaluate_backend_score(backend::ADTypes.AutoSymbolics, n, sparsity, stiff, is_sp)
    # Symbolic is perfect but doesn't scale.
    # Symbolic is perfect but doesn't scale well beyond moderate sizes.
    if n > 20
        return -100.0
    else
        return 110.0 # Prioritize symbolic for small-medium systems
    end
end

function evaluate_backend_score(backend::AutoFiniteDiff, n, sparsity, stiff, is_sp)
    # FiniteDiff is the baseline. 
    return 10.0
end

#==============================================================================#
# Decision Rationale & Linear Solver Selection
#==============================================================================#

function generate_rationale(backend, n, sparsity, stiff)
    if backend === nothing
        return "None: No suitable AD backend available."
    elseif backend isa AutoForwardDiff && !(backend isa AutoSparse)
        return "ForwardDiff: Optimal dual-number performance for small-medium systems (n=$n)."
    elseif backend isa AutoEnzyme
        return "Enzyme: High-performance reverse-mode scaling for large dense systems (n=$n)."
    elseif backend isa AutoSparse
        return "Sparse AD: Exploiting $(round(sparsity*100, digits=2))% density for PDE-optimal scaling."
    elseif backend isa ADTypes.AutoSymbolics
        return "Symbolics: Exact analytical precision for small-kernel system."
    else
        return "Backend: $(typeof(backend))"
    end
end

function select_linear_solver_for_backend(ad_backend, n, sparsity, stiff)
    # Joint (AD + Linear) Selection
    if ad_backend isa AutoSparse
        if n > 5000
            # For massive systems, we move to iterative solvers
            return KrylovJL_GMRES()
        elseif n > 500
            return KLUFactorization()
        else
            return UMFPACKFactorization()
        end
    end
    
    # Dense fallbacks
    if n > 500
        return LUFactorization()
    else
        return nothing # Default SciML (typically LU/QR)
    end
end
