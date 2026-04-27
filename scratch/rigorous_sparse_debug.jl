using Frankenstein
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using ADTypes
using SparseDiffTools
# Explicitly use the same coloring algorithm as injected in the code
using SparseMatrixColorings

# Setup the log file
log_file = open("scratch/sparsity_debug_log.txt", "w")
function log_msg(msg)
    println(msg)
    println(log_file, msg)
    flush(log_file)
end

log_msg("Starting Rigorous Sparse Diagnostic Test (V2)")
log_msg("Timestamp: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time()))")

# 1. Helper to create 2D Laplacian sparsity pattern
function laplacian_2d_sparsity(N)
    I_idx = Int[]
    J_idx = Int[]
    for j in 1:N, i in 1:N
        idx = (j-1)*N + i
        push!(I_idx, idx); push!(J_idx, idx)
        if i > 1;  push!(I_idx, idx); push!(J_idx, idx-1); end
        if i < N;  push!(I_idx, idx); push!(J_idx, idx+1); end
        if j > 1;  push!(I_idx, idx); push!(J_idx, idx-N); end
        if j < N;  push!(I_idx, idx); push!(J_idx, idx+N); end
    end
    V_idx = ones(Float64, length(I_idx))
    return sparse(I_idx, J_idx, V_idx)
end

# 2. 2D Heat Equation setup
N = 10 
dx = 1.0 / (N + 1)
D = 0.1
c = D / dx^2
function heat2d!(du, u, p, t)
    for j in 1:N, i in 1:N
        idx = (j-1)*N + i
        left  = i > 1   ? u[idx-1] : 0.0
        right = i < N   ? u[idx+1] : 0.0
        down  = j > 1   ? u[idx-N] : 0.0
        up    = j < N   ? u[idx+N] : 0.0
        center = u[idx]
        du[idx] = c * (left + right + down + up - 4*center)
    end
end
u0 = zeros(N*N)
u0[div(N, 2) + N*div(N, 2)] = 1.0

# 3. Sparsity Audit
log_msg("--- Sparsity Audit ---")
jp = laplacian_2d_sparsity(N)
log_msg("jac_prototype size: $(size(jp))")
log_msg("jac_prototype nnz: $(nnz(jp))")

# Numerical detection
ad_func! = (du, u) -> heat2d!(du, u, nothing, 0.0)
log_msg("Detecting sparsity numerically via SparseDiffTools...")
detected_jp = try
    SparseDiffTools.jacobian_sparsity(ad_func!, similar(u0), u0)
catch e
    log_msg("SparseDiffTools detection failed: $e")
    nothing
end

if detected_jp !== nothing
    log_msg("Detected JP nnz: $(nnz(detected_jp))")
    mismatch = nnz(jp) != nnz(detected_jp) || findnz(jp)[1:2] != findnz(detected_jp)[1:2]
    log_msg("Prototype vs Detected Mismatch: $mismatch")
end

# 4. Cache Creation Audit
log_msg("--- Cache Creation Audit ---")
# Use the default coloring if the specific one fails to dispatch
colorvec = try
    matrix_colors(jp)
catch e
    log_msg("Default coloring failed: $e")
    ones(Int, size(jp, 2))
end
log_msg("Number of colors (default): $(maximum(colorvec))")

# Try to create ForwardDiffColorJacobianCache
log_msg("Creating ForwardDiffColorJacobianCache with Prototype...")
try
    cache = ForwardDiffColorJacobianCache(ad_func!, u0, colorvec=colorvec, sparsity=jp)
    log_msg("Cache created successfully with jp.")
catch e
    log_msg("Cache creation FAILED with jp: $e")
end

# 5. Run Frankenstein and catch the error explicitly
log_msg("--- Frankenstein Run Audit ---")
# We setup Frankenstein to use EXACTLY what we want
f = ODEFunction(heat2d!, jac_prototype=jp)
prob = ODEProblem(f, u0, (0.0, 0.1))

# Manually trigger the "Injection" logic that happens in algorithm_selector.jl
# to see if it causes headers of the Error.
# We mimic the logic from MonsterSolver.jl

try
    # Force a specific sequence that usually triggers the bug
    sol = Frankenstein.solve(prob, Frankenstein.FrankensteinSolver())
    log_msg("Solve finished. Retcode: $(sol.retcode)")
catch e
    log_msg("SOLVE FAILED!")
    log_msg("Error type: $(typeof(e))")
    log_msg("Error message: $(sprint(showerror, e))")
end

close(log_file)
log_msg("Diagnostic complete. Results in scratch/sparsity_debug_log.txt")
