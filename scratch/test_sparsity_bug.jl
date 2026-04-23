using Frankenstein
using DifferentialEquations
using SparseArrays

println("Starting Sparsity Bug Reproduction (Fixed Version)...")

# A simple sparse system where we DON'T provide the jac_prototype
# but the system is large enough that Frankenstein might detect sparsity.
N = 100 
function sparse_f!(du, u, p, t)
    for i in 1:N
        du[i] = -u[i]
        if i > 1
            du[i] += 0.1 * u[i-1]
        end
        if i < N
            du[i] += 0.1 * u[i+1]
        end
    end
end
u0 = ones(N)
tspan = (0.0, 1.0)

# PROBLEM: No jac_prototype provided.
prob = ODEProblem(sparse_f!, u0, tspan)

println("Initializing solve with Frankenstein...")
try
    # We use Frankenstein.solve or just solve if exported
    sol = Frankenstein.solve(prob, Frankenstein.FrankensteinSolver())
    println("Solve successful! Retcode: ", sol.retcode)
    println("Final u norm: ", sum(abs.(sol.u[end])))
catch e
    println("Caught Error:")
    showerror(stdout, e)
    println()
    rethrow(e)
end
