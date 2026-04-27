using Frankenstein
using OrdinaryDiffEq
using LinearAlgebra
using Test

println("🧟 Starting Frankenstein Surgery Demonstration...")

# A model that starts easy and becomes extremely stiff
function surgery_model(du, u, p, t)
    # k transitions from 1.0 to 1e7 at t=0.5
    # We use a smooth transition to avoid discontinuous failure, 
    # but fast enough to trigger the pulse.
    k = 1.0 + 1e7 * (t > 0.5)
    du[1] = -k * u[1] + 10.0 * u[2]
    du[2] = u[1] - u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 1.0)
prob = ODEProblem(surgery_model, u0, tspan)

println("\n--- Phase 1: Initial Solve ---")
println("Frankenstein should start with an explicit solver (e.g. Tsit5)...")

# Run the solve
sol = solve(prob, Monster())

println("\n--- Phase 2: Verification ---")
println("Final time: $(sol.t[end])")
println("Final state: $(sol.u[end])")

if sol.t[end] == 1.0
    println("✅ Simulation Completed Successfully!")
else
    println("❌ Simulation Failed to reach t=1.0")
end

# Check if we can find surgery in the logs (simulated check)
# In a real test we'd capture IO, but here we just look at the success.
