using Frankenstein
using DifferentialEquations
using Test
using Logging

# Set up logging to see the monster's thoughts
global_logger(ConsoleLogger(stderr, Logging.Info))

@testset "Frankenstein Meta-Solver Adaptation & Retry" begin

    # 1. Van der Pol Oscillator - Stiff transition
    function vanderpol!(du, u, p, t)
        μ = p[1]
        du[1] = u[2]
        du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
    end

    u0 = [2.0, 0.0]
    tspan = (0.0, 100.0)
    p = [100.0] # Moderately stiff
    prob = ODEProblem(vanderpol!, u0, tspan, p)

    @info "Starting Van der Pol test with meta-solver..."
    
    # We use a custom Frankenstein instance to ensure we trigger the new logic
    monster = FrankensteinSolver()
    
    sol = Frankenstein.solve(prob, monster, reltol=1e-6, abstol=1e-8)
    
    @test sol.retcode == ReturnCode.Success
    @info "Solver finished with $(length(sol.t)) steps."

    # 2. Forced Instability Test
    p_extreme = [1e4] 
    prob_extreme = ODEProblem(vanderpol!, u0, (0.0, 1.0), p_extreme)
    
    @info "Testing instability retry logic with extreme stiffness..."
    sol_retry = Frankenstein.solve(prob_extreme, monster, reltol=1e-2, abstol=1e-2)
    
    @test sol_retry.retcode == ReturnCode.Success
    @info "Extreme test finished with $(length(sol_retry.t)) steps."

end
