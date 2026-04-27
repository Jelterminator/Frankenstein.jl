using Frankenstein
using SciMLBase

println("--- Debugging Oregonator (Truncation-Proof) ---")
function oregonator!(du, u, p, t)
    s, w, q = 77.27, 0.161, 8.375e-6
    du[1] = s*(u[2] + u[1]*(1.0 - q*u[1] - u[2]))
    du[2] = (u[3] - (1.0 + u[1])*u[2]) / s
    du[3] = w*(u[1] - u[3])
end
u0 = Float64[1.0, 2.0, 3.0]
p = Float64[]
tspan = (0.0, 2.0)
prob = ODEProblem(oregonator!, u0, tspan, p)

try
    sol = solve(prob, Monster(), max_retries=0)
    println("Success!")
catch e
    st = stacktrace(catch_backtrace())
    println("!!! ERROR SUMMARY !!!")
    println("REPEATED FOR TRUNCATION SAFETY:")
    for _ in 1:20
        println("ERROR: ", e)
    end
    println("-----------------------")
    Base.show_backtrace(stdout, catch_backtrace())
end
