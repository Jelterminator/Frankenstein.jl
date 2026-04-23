using Frankenstein
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Test

# ---------------------------------------------------------
# Test Problems (Defined at top level for visibility)
# ---------------------------------------------------------

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8/3) * u[3]
end

function vanderpol!(du, u, p, t)
    μ = p[1]
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
end

function rober!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = 0.04, 3e7, 1e4
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] =  k₁*y₁ - k₂*y₂^2 - k₃*y₂*y₃
    du[3] =  k₂*y₂^2
end

function heat_eq!(du, u, p, t)
    N = length(u)
    dx = 1.0 / (N + 1)
    c = p[1] / dx^2
    
    du[1] = c * (-2u[1] + u[2])
    for i in 2:N-1
        du[i] = c * (u[i-1] - 2u[i] + u[i+1])
    end
    du[N] = c * (u[N-1] - 2u[N])
end

@testset "Frankenstein.jl Comprehensive Test Suite" begin

    @testset "1. Non-Stiff & Mildly Stiff Regimes" begin
        u0_l = [1.0, 0.0, 0.0]
        tspan_l = (0.0, 10.0)
        prob_l = ODEProblem(lorenz!, u0_l, tspan_l)
        
        @info "Testing Frankenstein on Lorenz (Non-Stiff)"
        sol_l = Frankenstein.solve(prob_l, FrankensteinSolver())
        @test sol_l.retcode == ReturnCode.Success
        @test length(sol_l.t) > 0

        u0_v = [2.0, 0.0]
        tspan_v = (0.0, 20.0)
        p_v_mild = [10.0]
        prob_v_mild = ODEProblem(vanderpol!, u0_v, tspan_v, p_v_mild)

        @info "Testing Frankenstein on Van der Pol (μ=10, Mildly Stiff)"
        sol_v_mild = Frankenstein.solve(prob_v_mild, FrankensteinSolver())
        @test sol_v_mild.retcode == ReturnCode.Success
    end

    @testset "2. Highly Stiff Regimes" begin
        p_v_stiff = [1000.0]
        prob_v_stiff = ODEProblem(vanderpol!, [2.0, 0.0], (0.0, 3000.0), p_v_stiff)
        
        @info "Testing Frankenstein on Van der Pol (μ=1000, Highly Stiff)"
        sol_v_stiff = Frankenstein.solve(prob_v_stiff, FrankensteinSolver())
        @test sol_v_stiff.retcode == ReturnCode.Success

        prob_rober = ODEProblem(rober!, [1.0, 0.0, 0.0], (0.0, 100.0))
        
        @info "Testing Frankenstein on ROBER (Extreme Stiffness)"
        sol_rober = Frankenstein.solve(prob_rober, FrankensteinSolver())
        @test sol_rober.retcode == ReturnCode.Success
        @test sum(sol_rober.u[end]) ≈ 1.0 atol=1e-4 
    end

    @testset "3. Large & Sparse Systems" begin
        N_large = 200 
        u0_heat = zeros(N_large)
        u0_heat[Int(N_large/2)] = 1.0 
        p_heat = [0.1] 
        
        jac_sparsity = Tridiagonal(ones(N_large-1), -2ones(N_large), ones(N_large-1))
        f_sparse = ODEFunction(heat_eq!; jac_prototype=sparse(jac_sparsity))
        
        prob_heat = ODEProblem(f_sparse, u0_heat, (0.0, 10.0), p_heat)
        
        @info "Testing Frankenstein on Large Sparse 1D PDE (N=$N_large)"
        sol_heat = Frankenstein.solve(prob_heat, FrankensteinSolver())
        @test sol_heat.retcode == ReturnCode.Success
        @test length(sol_heat.u[end]) == N_large
    end

    @testset "Additional PDE Suite" begin
        include("test_pde_suite.jl")
    end

    @testset "4. Meta-Solver Preferences & Extreme Tolerances" begin
        u0_l = [1.0, 0.0, 0.0]
        prob_l = ODEProblem(lorenz!, u0_l, (0.0, 10.0))

        @info "Testing Memory Preference"
        sol_mem = Frankenstein.solve(prob_l, FrankensteinSolver(); prefer_memory=true)
        @test sol_mem.retcode == ReturnCode.Success

        @info "Testing Stability Preference"
        sol_stab = Frankenstein.solve(prob_l, FrankensteinSolver(); prefer_stability=true)
        @test sol_stab.retcode == ReturnCode.Success

        @info "Testing High Accuracy (1e-12)"
        sol_tight = Frankenstein.solve(prob_l, FrankensteinSolver(); reltol=1e-12, abstol=1e-12)
        @test sol_tight.retcode == ReturnCode.Success
    end

end