# backend_selector.jl
"""
    backend_selector.jl
    
Intelligent backend selection based on problem characteristics.
"""

using ADTypes
using LinearAlgebra
using SparseArrays

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
                        is_external_solver::Bool=false,
                        blacklist::Vector{String}=String[])
    
    # Filter out blacklisted backends
    filtered_backends = filter(b -> !(string(typeof(b)) in blacklist), available_backends)

    best_backend  = nothing
    best_score    = -Inf
    best_rationale = ""

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

    if is_external_solver
        best_backend = AutoFiniteDiff()
        best_score = 0.0
        best_rationale = "External Solver: Using built-in methods."
    else
        # Diagnostic: List all scores
        @info "[Frankenstein Selector] Evaluating scores for $n-var system (Sparse: $is_sp)"
        for backend in filtered_backends
            score = evaluate_backend_score(backend, n, sparsity, stiff, is_sp)
            @info "  - $(typeof(backend)): Score = $score"
            if score > best_score
                best_score     = score
                best_backend   = backend
                best_rationale = generate_rationale(backend, n, sparsity, stiff)
            end
        end
    end

    lin_solver = select_linear_solver_for_backend(best_backend, n, sparsity, stiff)
    return BackendSelection(best_backend, lin_solver, best_score, best_rationale)
end

function evaluate_backend_score(backend::AbstractADType, problem_size::Int, 
                                sparsity_ratio::Float64, is_stiff::Bool, is_sparse::Bool)
    score = 0.0
    b_str = string(typeof(backend))
    
    # NEW: Deconstructed string matching to handle wrapped types like AutoSparse{AutoForwardDiff}
    if occursin("AutoSparse", b_str) && occursin("ForwardDiff", b_str)
        if is_sparse
            score += 500.0
        else
            score -= 200.0 
        end
    elseif occursin("AutoEnzyme", b_str)
        score += (problem_size >= 250 ? 50.0 : 0.0)
    elseif occursin("AutoForwardDiff", b_str)
        score += (problem_size <= 150 ? 60.0 : 10.0)
    elseif occursin("AutoSymbolics", b_str) || occursin("AutoSymbolic", b_str)
        score += (problem_size <= 20 ? 100.0 : -50.0)
    elseif occursin("AutoFiniteDiff", b_str)
        score += 1.0 
    end
    
    return score
end

function generate_rationale(backend::AbstractADType, n::Int, 
                           sparsity::Float64, is_stiff::Bool)
    b_str = string(typeof(backend))
    if occursin("AutoForwardDiff", b_str) && !occursin("AutoSparse", b_str)
        return "ForwardDiff: Optimal for small-medium systems (n=$n)."
    elseif occursin("AutoEnzyme", b_str)
        return "Enzyme: Selected for high-performance reverse-mode AD on large system (n=$n)."
    elseif occursin("AutoSparse", b_str) && occursin("ForwardDiff", b_str)
        return "Sparse ForwardDiff: leveraging sparsity pattern ($(round(sparsity*100, digits=1))% density) for performance."
    elseif occursin("AutoSymbolics", b_str) || occursin("AutoSymbolic", b_str)
        return "Symbolics: Using exact analytical derivatives for small-scale precision."
    elseif occursin("AutoFiniteDiff", b_str)
        return "FiniteDiff: Robust numerical differentiation for non-standard code."
    else
        return "AD Backend: $(typeof(backend))"
    end
end

function select_linear_solver_for_backend(ad_backend, n, sparsity, stiff)
    # Basic logic for linear solver selection
    if n > 500
        return nothing # Let the solver pick UMFPACK/KLU automatically or use SciML default
    else
        return nothing
    end
end
