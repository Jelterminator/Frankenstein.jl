# backend_selector.jl
"""
    backend_selector.jl
    
Intelligent backend selection based on problem characteristics.
"""

using ADTypes
using LinearAlgebra

"""
    BackendSelection

Structure containing selected backends and their configurations.
"""
struct BackendSelection{AD, LS}
    ad_backend::AD
    linear_solver::LS
    performance_score::Float64
    selection_rationale::String
end

"""
    choose_backend(analysis::SystemAnalysis{T}, 
                        available_backends::Vector{AbstractADType}) where T

Choose the best backend combination based on problem characteristics.
"""
function choose_backend(analysis::SystemAnalysis{T}, 
                        available_backends::Vector{<:AbstractADType}) where T

    best_backend  = nothing
    best_score    = -Inf
    best_rationale = ""

    # Extract all needed inputs from the analysis object
    n       = analysis.system_size
    is_sp   = analysis.is_sparse
    
    # Safely compute sparsity ratio
    sparsity = 1.0 # Default dense
    if is_sp && analysis.sparsity_pattern !== nothing
        try
            sparsity = nnz(analysis.sparsity_pattern) / (n^2)
        catch
            sparsity = 0.1 # Heuristic fallback
        end
    elseif !is_sp
        sparsity = 1.0
    end
    
    stiff = analysis.is_stiff || (!isnan(analysis.stiffness_ratio) && analysis.stiffness_ratio > 1000.0)

    for backend in available_backends
        score = evaluate_backend_score(backend, n, sparsity, stiff)
        if score > best_score
            best_score     = score
            best_backend   = backend
            best_rationale = generate_rationale(backend, n, sparsity, stiff)
        end
    end

    # Pick the best linear solver for that backend
    lin_solver = select_linear_solver_for_backend(best_backend, n, sparsity, stiff)

    return BackendSelection(best_backend, lin_solver, best_score, best_rationale)
end

function select_linear_solver_for_backend(ad_backend, n, sparsity, stiff)
    if n > 1000
        if sparsity < 0.1
            return KLUFactorization()
        else
            # Use a robust iterative solver for large non-sparse systems
            return :GMRES # Symbol dispatch is more robust across modules
        end
    elseif n > 200
        return UMFPACKFactorization()
    elseif n > 50
        # For medium systems, use a standard LU
        return LUFactorization()
    else
        return nothing # Default dense (usually internal LU)
    end
end


"""
    evaluate_backend_score(backend, problem_size, sparsity_ratio, is_stiff)

Score a backend based on problem characteristics.
"""
function evaluate_backend_score(backend::AbstractADType, problem_size::Int, 
                                sparsity_ratio::Float64, is_stiff::Bool)
    score = 0.0
    
    # Size-based scoring
    b_name = name(backend)
    
    if occursin("AutoForwardDiff", b_name)
        score += problem_size <= 100 ? 12.0 : max(0.0, 10.0 - problem_size/50)
    elseif occursin("AutoEnzyme", b_name)
        score += problem_size >= 50 ? 15.0 : max(0.0, problem_size/5)
        if is_stiff; score += 5.0; end
    elseif occursin("AutoSparseForwardDiff", b_name)
        score += (sparsity_ratio < 0.1 && problem_size > 100) ? 20.0 : 5.0
    elseif occursin("AutoSymbolic", b_name)
        score += problem_size <= 50 ? 8.0 : -10.0 # Bad for large systems
    elseif occursin("AutoFiniteDiff", b_name)
        score += 3.0  # Always available but slow
    elseif b_name == "HybridBackend" 
        score += 10.0 
    end
    
    # Sparsity bonus
    if (occursin("SparseForwardDiff", b_name) || (occursin("Enzyme", b_name) && problem_size > 500)) && sparsity_ratio < 0.1
        score += 5.0
    end


    
    return score
end

# Helper to identify HybridBackend without importing the module if it might be circular
name(b::AbstractADType) = string(typeof(b))

"""
    generate_rationale(backend, problem_size, sparsity_ratio, is_stiff)

Generate human-readable rationale for backend selection.
"""
function generate_rationale(backend::AbstractADType, problem_size::Int, 
                           sparsity_ratio::Float64, is_stiff::Bool)
    if backend isa AutoForwardDiff
        return "ForwardDiff: Optimal for small-medium systems (n=$problem_size)."
    elseif backend isa AutoEnzyme
        return "Enzyme: Selected for high-performance reverse-mode AD on large system (n=$problem_size)."
    elseif backend isa AutoSparseForwardDiff
        return "Sparse ForwardDiff: Leveraging sparsity pattern ($(round(sparsity_ratio*100, digits=1))% density)."
    elseif backend isa AutoSymbolic
        return "Symbolics: Using analytical expression for small system (n=$problem_size)."
    elseif backend isa AutoFiniteDiff
        return "FiniteDiff: Robust fallback for complex code."
    elseif name(backend) == "HybridBackend"
        return "Hybrid: Multi-strategy backend for maximum stability."
    else
        return "Custom backend: $(typeof(backend))"
    end
end

