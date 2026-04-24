# algorithm_selector.jl - Trait-based algorithm selection and configuration
"""
    algorithm_selector.jl
    
Hardware-aware, trait-based algorithm configuration.
"""

module AlgorithmSelector

using SciMLBase
using ADTypes
using LinearSolve
using SparseMatrixColorings
using ..FCore
using ..Backends

using ..ExplicitSolvers: get_explicit_recommendations
using ..StiffSolvers: get_stiff_recommendations
using ..CompositeSolvers: get_composite_recommendations
using ..MultiscaleSolvers: get_multiscale_recommendations
using ..SparseSolvers: get_sparse_recommendations
using ..AdaptiveSolvers: get_adaptive_recommendations
using ..ParallelSolvers: get_parallel_recommendations
using ..SpecialtySolvers: get_specialty_recommendations

#==============================================================================#
# Unified Recommendation Interface
#==============================================================================#

function get_all_recommendations(analysis::SystemAnalysis; rtol::Float64=1e-6,
                                  prefer_memory::Bool=false,
                                  prefer_stability::Bool=true)
    # 1. Collect from all sub-modules with preference propagation
    recs = vcat(
        get_explicit_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_stiff_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_composite_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_multiscale_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_sparse_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_adaptive_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_parallel_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability),
        get_specialty_recommendations(analysis; rtol=rtol, prefer_memory=prefer_memory, prefer_stability=prefer_stability)
    )

    # 2. De-duplicate based on algorithm type
    # We keep the one with the highest priority if duplicates exist
    unique_recs = Dict{Any, AlgorithmRecommendation}()
    for rec in recs
        alg_key = typeof(rec.algorithm)
        if !haskey(unique_recs, alg_key) || rec.priority > unique_recs[alg_key].priority
            unique_recs[alg_key] = rec
        end
    end
    
    # 3. Final sorting
    final_list = collect(values(unique_recs))
    return sort(final_list; by = rec -> -compute_adjusted_priority(rec, analysis;
                                   prefer_memory=prefer_memory,
                                   prefer_stability=prefer_stability))
end

function select_algorithm(analysis::SystemAnalysis; kwargs...)
    # Get all candidate recommendations
    recs = get_all_recommendations(analysis)
    return isempty(recs) ? nothing : first(recs)
end

#==============================================================================#
# Hardware Awareness & Trait discovery
#==============================================================================#

"""
    is_gpu(u) -> Bool
Check if the state vector is on a GPU.
"""
function is_gpu(u)
    # Check for CuArray or similar GPU arrays without requiring the packages as dependencies
    u_type = string(typeof(u))
    return occursin("CuArray", u_type) || occursin("ROCArray", u_type) || occursin("GPUArray", u_type)
end

"""
    create_solver_configuration(...)
    
Instantiates an algorithm with AD and Linear Solver using trait-based discovery.
"""
function create_solver_configuration(rec, analysis, backend_selection;
                                     reltol::Float64=1e-6,
                                     abstol::Float64=1e-6,
                                     options...)
    
    alg_constructor = rec.algorithm
    ad_backend = backend_selection.ad_backend
    lin_solver = backend_selection.linear_solver
    u0 = analysis.jacobian !== nothing ? zeros(eltype(analysis.jacobian), analysis.system_size) : zeros(analysis.system_size)
    
    # 1. Hardware Awareness Override
    gpu_active = is_gpu(u0)
    if gpu_active
        @info "[Frankenstein] GPU detected! Optimizing configuration for device execution."
        # Note: In a future update, we might swap out CPU-bound linear solvers here.
    end

    # 2. Determine constructor capabilities
    # We use a robust property/method check instead of try-catch guessing
    instance_temp = try alg_constructor() catch; nothing end
    
    # Discovery of keywords supported by the algorithm
    supports_autodiff = false
    supports_linsolve = false
    
    if instance_temp !== nothing
        # Many SciML algorithms define these traits
        supports_autodiff = hasproperty(instance_temp, :autodiff) || occursin("autodiff", string(methods(alg_constructor)))
        supports_linsolve = hasproperty(instance_temp, :linsolve) || occursin("linsolve", string(methods(alg_constructor)))
    end

    # 3. Handle technical Workarounds
    # Rodas/Rosenbrock methods often prefer Bool autodiff in certain SciML versions
    final_ad = ad_backend
    alg_name = string(alg_constructor)
    if (occursin("Rodas", alg_name) || occursin("Rosenbrock", alg_name))
        if ad_backend isa AutoForwardDiff
            final_ad = true
        elseif ad_backend isa ADTypes.AutoSymbolics # FIX: Typo
            # Rosenbrock methods in OrdinaryDiffEq sometimes fail with NoFunctionWrapperFoundError
            # when passed AutoSymbolics() directly. Mapping to true (ForwardDiff)
            # avoids this while maintaining high performance. 
            # Future note: Symbolic J compilation should be done at the ODEFunction level.
            final_ad = true 
        end
    end

    # 3. Sparse Coloring & Pattern Injection
    final_ad = if ad_backend isa ADTypes.AutoSparse
        # If it's a sparse backend, we ensure it has a coloring algorithm.
        # We use AutoFiniteDiff as it is often more robust to minor sparsity mismatches
        # between detection and the internal cache.
        if (analysis.is_sparse || analysis.sparsity_pattern !== nothing)
            @info "[Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling."
            
            ADTypes.AutoSparse(ADTypes.AutoFiniteDiff(); 
                               sparsity_detector = Backends.PrecomputedSparsityDetector(analysis.sparsity_pattern),
                               coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
        else
            final_ad # FIX: Return final_ad instead of ad_backend
        end
    else
        final_ad # FIX: Return final_ad instead of ad_backend
    end

    # 4. Cohesive unit instantiation
    instance = if supports_autodiff && supports_linsolve
        alg_constructor(autodiff = final_ad, linsolve = lin_solver)
    elseif supports_autodiff
        alg_constructor(autodiff = final_ad)
    elseif supports_linsolve
        alg_constructor(linsolve = lin_solver)
    else
        alg_constructor()
    end
    
    return (algorithm = instance,
            reltol = reltol,
            abstol = abstol,
            options = options)
end

export get_all_recommendations, select_algorithm, create_solver_configuration

end # module AlgorithmSelector
