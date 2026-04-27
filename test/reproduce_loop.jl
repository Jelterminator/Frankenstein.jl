
using Frankenstein
using DifferentialEquations
using Test

function extreme_oregonator!(du, u, p, t)
    s = 77.27
    w = 0.161
    q = 8.375e-6
    du[1] = s * (u[2] + u[1] * (1 - w * u[1] - u[2]))
    du[2] = (u[3] - (1 + u[1]) * u[2]) / s
    du[3] = w * (u[1] - u[3])
end

u0 = [1.0, 2.0, 3.0]
tspan = (0.0, 0.5) # Short span to catch the initial transition
prob = ODEProblem(extreme_oregonator!, u0, tspan)

@info "--- Reproducing Surgery Loop ---"
sol = Frankenstein.monster_solve!(prob, Frankenstein.Monster(); reltol=1e-9, abstol=1e-12)

@test sol.retcode == SciMLBase.ReturnCode.Success
# Count how many times "Surgery Successful" appeared in the logs (manually or via a mock if we had one)
# For this script, we just want to see it finish quickly.
