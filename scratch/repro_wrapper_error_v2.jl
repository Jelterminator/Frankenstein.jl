using Frankenstein
using DifferentialEquations
using ADTypes
using SciMLBase

println("--- Testing if wrap_code=false fixes the issue ---")

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8/3) * u[3]
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1.0)

# Create ODEFunction with wrap_code=false
f_nowrap = ODEFunction(lorenz!, wrap_code=false)
prob = ODEProblem(f_nowrap, u0, tspan)

println("System size: ", length(u0))

solver = Frankenstein.FrankensteinSolver()
try
    # Still force AutoSymbolics
    sol = Frankenstein.solve(prob, solver, ad_available=[AutoSymbolics()])
    println("Solve successful!")
catch e
    println("CAUGHT ERROR:")
    showerror(stdout, e)
    println()
end
