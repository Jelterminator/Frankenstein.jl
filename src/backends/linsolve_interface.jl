# linsolve_interface.jl
"""
    linsolve_interface.jl
    
Enhanced interface for linear solver selection and configuration.
    
This module provides intelligent linear solver selection based on matrix properties,
problem characteristics, and performance considerations.
"""

using LinearSolve
using SparseArrays
using LinearAlgebra
using Preferences

# Optional dependencies with graceful fallbacks
const PARDISO_AVAILABLE = try
    using Pardiso
    true
catch
    false
end

const MUMPS_AVAILABLE = try
    using MUMPS
    true
catch
    false
end

const INCOMPLETELU_AVAILABLE = try
    using IncompleteLU
    true
catch
    false
end

#==============================================================================#
# Linear Solver Selection Types and Structures
#==============================================================================#

"""
    LinearSolverConfig{S, P}

Configuration structure containing the selected linear solver and preconditioner.
"""
struct LinearSolverConfig{S, P}
    solver::S
    preconditioner::P
    rationale::String
    expected_performance::Symbol  # :fast, :medium, :slow
end

"""
    MatrixProperties

Structure to encapsulate matrix properties for solver selection.
"""
struct MatrixProperties
    size::Int
    sparsity_ratio::Float64
    condition_number::Float64
    is_symmetric::Bool
    is_positive_definite::Bool
    is_diagonally_dominant::Bool
    bandwidth::Int
    nnz::Int
end

"""
    analyze_matrix(A)

Analyze matrix properties to inform solver selection.
"""
function analyze_matrix(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    
    # Basic properties
    is_sparse = issparse(A)
    nnz_count = is_sparse ? nnz(A) : n^2
    sparsity_ratio = 1.0 - (nnz_count / (n^2))
    
    # Symmetry check (efficient for sparse matrices)
    is_symmetric = issymmetric(A)
    
    # Positive definiteness (expensive, only for small matrices)
    is_positive_definite = false
    if n <= 100 && is_symmetric
        try
            cholesky(A)
            is_positive_definite = true
        catch
            is_positive_definite = false
        end
    end
    
    # Diagonal dominance check
    is_diagonally_dominant = check_diagonal_dominance(A)
    
    # Bandwidth estimation (for banded matrices)
    bandwidth = estimate_bandwidth(A)
    
    # Condition number estimation (expensive, only for small matrices)
    condition_number = n <= 200 ? cond(A) : NaN
    
    return MatrixProperties(
        n, sparsity_ratio, condition_number, is_symmetric, 
        is_positive_definite, is_diagonally_dominant, bandwidth, nnz_count
    )
end

"""
    check_diagonal_dominance(A)

Check if matrix is diagonally dominant.
"""
function check_diagonal_dominance(A::AbstractMatrix)
    n = size(A, 1)
    for i in 1:n
        row_sum = sum(abs, A[i, :]) - abs(A[i, i])
        if abs(A[i, i]) <= row_sum
            return false
        end
    end
    return true
end

"""
    estimate_bandwidth(A)

Estimate the bandwidth of a matrix.
"""
function estimate_bandwidth(A::AbstractMatrix)
    n = size(A, 1)
    if issparse(A)
        I, J, _ = findnz(A)
        return isempty(I) ? 0 : maximum(abs.(I .- J))
    else
        # For dense matrices, check for banded structure
        max_band = 0
        for i in 1:n, j in 1:n
            if abs(A[i, j]) > eps() * norm(A, Inf)
                max_band = max(max_band, abs(i - j))
            end
        end
        return max_band
    end
end

#==============================================================================#
# Enhanced Solver Selection
#==============================================================================#

"""
    select_linear_solver(A, b=nothing; problem_type=:general, 
                        performance_priority=:balanced, 
                        memory_constraints=false)

Select an appropriate linear solver based on matrix properties and constraints.

# Arguments
- `A`: The coefficient matrix
- `b`: The right-hand side vector (optional, used for additional analysis)
- `problem_type`: Type of problem (`:general`, `:stiff`, `:symmetric_pd`, `:saddle_point`)
- `performance_priority`: Priority (`:speed`, `:memory`, `:accuracy`, `:balanced`)
- `memory_constraints`: Whether to prioritize memory-efficient solvers

# Returns
- `LinearSolverConfig`: Configuration with solver, preconditioner, and rationale
"""
function select_linear_solver(A, b=nothing; 
                            problem_type=:general, 
                            performance_priority=:balanced,
                            memory_constraints=false)
    
    # Analyze matrix properties
    props = analyze_matrix(A)
    
    # Select solver based on properties and constraints
    solver_config = select_solver_by_properties(props, problem_type, 
                                               performance_priority, memory_constraints)
    
    return solver_config
end

"""
    select_solver_by_properties(props, problem_type, performance_priority, memory_constraints)

FCore solver selection logic based on matrix properties.
"""
function select_solver_by_properties(props::MatrixProperties, 
                                    problem_type::Symbol,
                                    performance_priority::Symbol,
                                    memory_constraints::Bool)
    
    n = props.size
    
    # Very small problems: direct methods
    if n <= 50
        return select_small_problem_solver(props, problem_type)
    end
    
    # Medium problems: balanced approach
    if n <= 1000
        return select_medium_problem_solver(props, problem_type, performance_priority)
    end
    
    # Large problems: iterative methods preferred
    return select_large_problem_solver(props, problem_type, performance_priority, memory_constraints)
end

"""
    select_small_problem_solver(props, problem_type)

Solver selection for small problems (n ≤ 50).
"""
function select_small_problem_solver(props::MatrixProperties, problem_type::Symbol)
    
    # Symmetric positive definite
    if props.is_positive_definite
        return LinearSolverConfig(
            CholeskyFactorization(),
            nothing,
            "Cholesky factorization for small symmetric positive definite matrix",
            :fast
        )
    end
    
    # Symmetric indefinite
    if props.is_symmetric
        return LinearSolverConfig(
            BunchKaufmanFactorization(),
            nothing,
            "Bunch-Kaufman factorization for small symmetric indefinite matrix",
            :fast
        )
    end
    
    # General small problem
    return LinearSolverConfig(
        LUFactorization(),
        nothing,
        "LU factorization for small general matrix",
        :fast
    )
end

"""
    select_medium_problem_solver(props, problem_type, performance_priority)

Solver selection for medium problems (50 < n ≤ 1000).
"""
function select_medium_problem_solver(props::MatrixProperties, 
                                    problem_type::Symbol,
                                    performance_priority::Symbol)
    
    # Sparse matrices
    if props.sparsity_ratio > 0.5
        return select_sparse_solver(props, problem_type, performance_priority)
    end
    
    # Dense matrices
    return select_dense_solver(props, problem_type, performance_priority)
end

"""
    select_large_problem_solver(props, problem_type, performance_priority, memory_constraints)

Solver selection for large problems (n > 1000).
"""
function select_large_problem_solver(props::MatrixProperties, 
                                   problem_type::Symbol,
                                   performance_priority::Symbol,
                                   memory_constraints::Bool)
    
    # Always prefer iterative methods for large problems
    if props.is_positive_definite
        precond = configure_preconditioner(props, :cg)
        return LinearSolverConfig(
            KrylovJL_CG(),
            precond,
            "Conjugate Gradient for large symmetric positive definite matrix",
            :medium
        )
    end
    
    # For general large problems, use GMRES
    precond = configure_preconditioner(props, :gmres)
    return LinearSolverConfig(
        KrylovJL_GMRES(),
        precond,
        "GMRES for large general matrix",
        :medium
    )
end

"""
    select_sparse_solver(props, problem_type, performance_priority)

Specialized solver selection for sparse matrices.
"""
function select_sparse_solver(props::MatrixProperties, 
                             problem_type::Symbol,
                             performance_priority::Symbol)
    
    n = props.size
    
    # Very sparse matrices with good sparsity pattern
    if props.sparsity_ratio > 0.9
        if PARDISO_AVAILABLE && performance_priority == :speed
            return LinearSolverConfig(
                PardisoJL(),
                nothing,
                "PARDISO for very sparse matrix (high performance)",
                :fast
            )
        elseif MUMPS_AVAILABLE
            return LinearSolverConfig(
                MUMPSFactorization(),
                nothing,
                "MUMPS for very sparse matrix",
                :fast
            )
        else
            return LinearSolverConfig(
                KLUFactorization(),
                nothing,
                "KLU factorization for sparse matrix",
                :fast
            )
        end
    end
    
    # Moderately sparse matrices
    if n <= 5000
        return LinearSolverConfig(
            KLUFactorization(),
            nothing,
            "KLU factorization for moderately sparse matrix",
            :fast
        )
    else
        # Large sparse: use iterative
        precond = configure_preconditioner(props, :gmres)
        return LinearSolverConfig(
            KrylovJL_GMRES(),
            precond,
            "GMRES with preconditioning for large sparse matrix",
            :medium
        )
    end
end

"""
    select_dense_solver(props, problem_type, performance_priority)

Specialized solver selection for dense matrices.
"""
function select_dense_solver(props::MatrixProperties, 
                            problem_type::Symbol,
                            performance_priority::Symbol)
    
    n = props.size
    
    # Symmetric positive definite
    if props.is_positive_definite
        return LinearSolverConfig(
            CholeskyFactorization(),
            nothing,
            "Cholesky factorization for dense symmetric positive definite matrix",
            :fast
        )
    end
    
    # Symmetric indefinite
    if props.is_symmetric
        return LinearSolverConfig(
            BunchKaufmanFactorization(),
            nothing,
            "Bunch-Kaufman factorization for dense symmetric indefinite matrix",
            :fast
        )
    end
    
    # General dense matrix
    if n <= 500
        return LinearSolverConfig(
            LUFactorization(),
            nothing,
            "LU factorization for medium dense matrix",
            :fast
        )
    else
        # Large dense: consider iterative if condition number is reasonable
        if !isnan(props.condition_number) && props.condition_number < 1e12
            return LinearSolverConfig(
                KrylovJL_GMRES(),
                configure_preconditioner(props, :gmres),
                "GMRES for large dense matrix with reasonable condition number",
                :medium
            )
        else
            return LinearSolverConfig(
                LUFactorization(),
                nothing,
                "LU factorization for large dense matrix (ill-conditioned)",
                :slow
            )
        end
    end
end

#==============================================================================#
# Enhanced Preconditioner Configuration
#==============================================================================#

"""
    configure_preconditioner(props::MatrixProperties, solver_type::Symbol)

Configure an appropriate preconditioner based on matrix properties and solver type.
"""
function configure_preconditioner(props::MatrixProperties, solver_type::Symbol)
    
    # No preconditioning for small problems
    if props.size <= 100
        return nothing
    end
    
    # Diagonal preconditioning for diagonally dominant matrices
    if props.is_diagonally_dominant
        return DiagonalPreconditioner()
    end
    
    # Specialized preconditioners based on solver type
    if solver_type == :cg
        return configure_cg_preconditioner(props)
    elseif solver_type == :gmres
        return configure_gmres_preconditioner(props)
    elseif solver_type == :bicgstab
        return configure_bicgstab_preconditioner(props)
    end
    
    return nothing
end

"""
    configure_cg_preconditioner(props)

Configure preconditioner for Conjugate Gradient method.
"""
function configure_cg_preconditioner(props::MatrixProperties)
    # For CG, we need symmetric positive definite preconditioners
    if props.sparsity_ratio > 0.5
        return IncompleteCholesky()  # For sparse SPD matrices
    else
        return DiagonalPreconditioner()  # Simple diagonal scaling
    end
end

"""
    configure_gmres_preconditioner(props)

Configure preconditioner for GMRES method.
"""
function configure_gmres_preconditioner(props::MatrixProperties)
    if props.sparsity_ratio > 0.5
        if INCOMPLETELU_AVAILABLE
            return IncompleteLUPreconditioner()
        else
            return DiagonalPreconditioner()
        end
    else
        return DiagonalPreconditioner()
    end
end

"""
    configure_bicgstab_preconditioner(props)

Configure preconditioner for BiCGSTAB method.
"""
function configure_bicgstab_preconditioner(props::MatrixProperties)
    # Similar to GMRES but can handle more general preconditioners
    return configure_gmres_preconditioner(props)
end

#==============================================================================#
# Preconditioner Types (Placeholders for actual implementations)
#==============================================================================#

"""
    DiagonalPreconditioner

Simple diagonal (Jacobi) preconditioner.
"""
struct DiagonalPreconditioner end

"""
    IncompleteCholesky

Incomplete Cholesky preconditioner for symmetric positive definite matrices.
"""
struct IncompleteCholesky end

"""
    IncompleteLUPreconditioner

Incomplete LU preconditioner for general matrices.
"""
struct IncompleteLUPreconditioner end

#==============================================================================#
# Solver Performance Optimization
#==============================================================================#

"""
    optimize_solver_parameters(solver, props::MatrixProperties)

Optimize solver parameters based on matrix properties.
"""
function optimize_solver_parameters(solver, props::MatrixProperties)
    # This would contain solver-specific parameter optimization
    # For now, return the solver as-is
    return solver
end

"""
    estimate_solve_time(solver_config::LinearSolverConfig, props::MatrixProperties)

Estimate the time complexity for solving with the given configuration.
"""
function estimate_solve_time(solver_config::LinearSolverConfig, props::MatrixProperties)
    n = props.size
    
    if solver_config.solver isa LUFactorization
        return :O_n3  # O(n³) complexity
    elseif solver_config.solver isa CholeskyFactorization
        return :O_n3_div2  # O(n³/2) complexity
    elseif solver_config.solver isa KrylovJL_CG
        return :O_n2_iter  # O(n²) per iteration
    elseif solver_config.solver isa KrylovJL_GMRES
        return :O_n2_iter  # O(n²) per iteration
    elseif solver_config.solver isa KLUFactorization
        return :O_nnz  # O(nnz) for sparse
    else
        return :unknown
    end
end

"""
    validate_solver_choice(solver_config::LinearSolverConfig, props::MatrixProperties)

Validate that the chosen solver is appropriate for the matrix properties.
"""
function validate_solver_choice(solver_config::LinearSolverConfig, props::MatrixProperties)
    solver = solver_config.solver
    
    # Check for obvious mismatches
    if solver isa CholeskyFactorization && !props.is_positive_definite
        @warn "Cholesky factorization selected for non-positive definite matrix"
        return false
    end
    
    if solver isa KrylovJL_CG && !props.is_positive_definite
        @warn "Conjugate Gradient selected for non-positive definite matrix"
        return false
    end
    
    if props.size > 10000 && solver isa LUFactorization
        @warn "Direct LU factorization selected for very large matrix (consider iterative methods)"
        return false
    end
    
    return true
end

#==============================================================================#
# Utility Functions
#==============================================================================#

"""
    get_solver_info(solver_config::LinearSolverConfig)

Get human-readable information about the solver configuration.
"""
function get_solver_info(solver_config::LinearSolverConfig)
    return """
    Solver: $(typeof(solver_config.solver))
    Preconditioner: $(solver_config.preconditioner === nothing ? "None" : typeof(solver_config.preconditioner))
    Rationale: $(solver_config.rationale)
    Expected Performance: $(solver_config.expected_performance)
    """
end

"""
    benchmark_solver(solver_config::LinearSolverConfig, A, b)

Benchmark the solver configuration on the given problem.
"""
function benchmark_solver(solver_config::LinearSolverConfig, A, b)
    # This would implement actual benchmarking
    # For now, return placeholder metrics
    return (
        setup_time = 0.0,
        solve_time = 0.0,
        memory_usage = 0,
        iterations = 0,
        residual = 0.0
    )
end
