println("JULIA_SCRIPT_START")
using Frankenstein
println("FRANKENSTEIN_LOADED")
using DifferentialEquations
println("DIFFEQ_LOADED")
using ADTypes
println("ADTYPES_LOADED")

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8/3) * u[3]
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)

println("SYSTEM_INITIALIZED")
solver = Frankenstein.FrankensteinSolver()
sol = Frankenstein.solve(prob, solver, ad_available=[AutoSymbolics()])
println("SOLVE_FINISHED")
@show sol.retcode
