# analysis.jl - Core system analysis functionality

module Analysis

using SciMLBase
using LinearAlgebra
using SparseArrays
using Statistics

using ..FCore
using ..Utilities.Jacobians
include("sparsity_analysis.jl")
using .Sparsity: detect_sparsity_patterns

export analyze_system_structure, 
       initial_stiffness_estimate,
       initial_timescale_analysis,
       estimate_coupling_strength,
       analyze_sparsity,
       check_stability_region,
       needs_analysis_update!

"""
    analyze_system_structure(prob::SciMLBase.ODEProblem) -> SystemAnalysis
"""
function analyze_system_structure(prob)
    u0 = prob.u0
    t0 = prob.tspan[1]
    p = prob.p
    f = prob.f

    # 1. Compute system size and detect sparsity patterns
    system_size = length(u0)
    sparsity = detect_sparsity_patterns(prob)
    
    # 2. Natural, data-driven sparsity check (Threshold 35%)
    density = (sparsity !== nothing) ? (nnz(sparsity) / (system_size^2)) : 1.0
    is_sparse = (sparsity !== nothing) && (density < 0.35)
    
    @info "[Frankenstein Analysis] System Size: $system_size | Sparse: $is_sparse | Density: $(round(density*100, digits=2))%"
    
    # 3. Compute Jacobian with fallback
    local J
    try
        J = compute_jacobian(f, u0, p, t0)
    catch e
        @warn "Jacobian computation failed: $e. Using finite differences."
        J = finite_difference_jac(f, u0, p, t0, inplace=SciMLBase.isinplace(f))
    end
    J = is_sparse ? sparse(J) : J

    # 4. Perform further analyses
    stiffness = initial_stiffness_estimate(f, u0, p, J0=J)
    timescales = initial_timescale_analysis(J)
    coupling = estimate_coupling_strength(J)
    condition = cond(collect(J)) # Use dense for cond if small, or estimate
    
    return SystemAnalysis{Float64}(
        stiffness, stiffness > 1e4, sparsity, timescales, coupling, condition, 
        system_size, is_sparse, J, 0, 0, 0, 0.0, 0, Any[]
    )
end

function initial_stiffness_estimate(f, u, p; J0=nothing)
    # Spectral radius based estimate
    evals = eigvals(collect(J0))
    radius = maximum(abs.(evals))
    return radius # Return raw spectral radius as a proxy for stiffness
end

function initial_timescale_analysis(J)
    evals = eigvals(collect(J))
    return abs.(1.0 ./ evals)
end

function estimate_coupling_strength(J)
    # Using L1 norm of off-diagonal elements as a crude measure of coupling
    n = size(J, 1)
    if n <= 1; return 0.0; end
    
    total = sum(abs.(J))
    diag_total = sum(abs.(diag(J)))
    return (total - diag_total) / (n*(n-1))
end

function check_stability_region(sol, stiffness)
    # Dynamic check based on current dt and stiffness
    return sol.dt * stiffness < 2.0
end

function needs_analysis_update!(analysis, integrator)
    # Heuristic for when to re-analyze
    return integrator.nsteps - analysis.last_update_step > 50
end

end # module Analysis
