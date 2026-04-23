# backends.jl
module Backends

using ADTypes
using LinearSolve
using SparseArrays
using ForwardDiff
using FiniteDiff
using Symbolics
using SparseDiffTools
using ..FCore  # Access to FCore types

# Include backend files
include("AD_interface.jl")
include("linsolve_interface.jl") 
include("sparse_forwarddiff.jl")
include("enzyme_backend.jl")
include("finite_difference.jl")
include("symbolic_backend.jl")
include("hybrid_backend.jl")
include("backend_selector.jl")

# Export key functions
export jacobian, select_ad_backend, select_linear_solver
export configure_preconditioner
export configure_sparse_forwarddiff, configure_enzyme, configure_finite_diff, configure_symbolic

export switch_backend, choose_backend
export BackendPerformanceMetrics, evaluate_backend_performance
export AutoSymbolic


#==============================================================================#
# Backend Performance Metrics
#==============================================================================#

"""
    BackendPerformanceMetrics{T}

Stores performance metrics for different AD backends to aid in selection.
"""
mutable struct BackendPerformanceMetrics{T}
    jacobian_time::T
    memory_allocation::Int
    accuracy::T
    sparsity_efficiency::T
    stability::T
end

function BackendPerformanceMetrics{T}() where T
    return BackendPerformanceMetrics{T}(T(Inf), 0, T(0), T(0), T(0))
end

end # module Backends











