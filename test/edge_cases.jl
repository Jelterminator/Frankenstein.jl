using Frankenstein
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Test

# Helper to create tridiagonal sparsity pattern
function tridiagonal_sparsity(N)
    I_idx = Int[]
    J_idx = Int[]
    for i in 1:N
        push!(I_idx, i); push!(J_idx, i)
        if i > 1; push!(I_idx, i); push!(J_idx, i-1); end
        if i < N; push!(I_idx, i); push!(J_idx, i+1); end
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)))
end

# Helper for consistent benchmarking
function compare_edge_cases(prob, name; manual_sciml=nothing)
    @info "=== Benchmarking Edge Case: $name ==="
    
    # 1. Standard SciML (PolyAlgorithm)
    t_sciml = @elapsed sol_sciml = solve(prob)
    
    # 2. Frankenstein Monster
    t_monster = @elapsed sol_monster = solve(prob, Monster())
    
    println("  [SciML Default] Time: $(round(t_sciml, digits=4))s | Steps: $(length(sol_sciml.t)) | Retcode: $(sol_sciml.retcode)")
    println("  [Frankenstein]  Time: $(round(t_monster, digits=4))s | Steps: $(length(sol_monster.t)) | Retcode: $(sol_monster.retcode)")
    
    if sol_monster.retcode == ReturnCode.Success && (t_monster < t_sciml || length(sol_monster.t) < length(sol_sciml.t))
        println("  >> ADVANTAGE: Frankenstein reduced steps or time.")
    end
    
    println("------------------------------------------")
    return sol_monster, sol_sciml
end

@testset "Frankenstein Edge Cases" begin

    @testset "1. Emergency Recovery (The 'Monster' Survival)" begin
        # System that becomes extremely stiff suddenly.
        # Most explicit solvers will fail.
        function explosive!(du, u, p, t)
            k = t < 0.1 ? 1.0 : 1e8
            du[1] = -k * u[1] + 10.0
        end
        u0 = [1.0]
        tspan = (0.0, 0.2)
        prob = ODEProblem(explosive!, u0, tspan)
        
        @info "--- Testing Emergency Recovery ---"
        # Tsit5 will fail.
        sol_tsit = solve(prob, Tsit5(), maxiters=1000)
        println("  [Tsit5] Retcode: $(sol_tsit.retcode) (Expected: MaxIters or Failure)")
        
        t_monster = @elapsed sol_monster = solve(prob, Monster())
        println("  [Monster] Time: $(round(t_monster, digits=4))s | Retcode: $(sol_monster.retcode) (Expected: Success)")
        
        @test sol_monster.retcode == ReturnCode.Success
        @test sol_tsit.retcode != ReturnCode.Success
    end

    @testset "2. The Sparsity-Aware Pivot (N=2000)" begin
        # Large system where picking Sparse AD + KLU is critical.
        N = 2000
        dx = 1.0/N
        D = 1e-1
        function large_sparse!(du, u, p, t)
            k = 1e3 * (1 + tanh(50*(t - 0.1))) / 2
            for i in 1:N
                left = i > 1 ? u[i-1] : u[N]
                right = i < N ? u[i+1] : u[1]
                du[i] = D * (left - 2*u[i] + right) / dx^2 - k * u[i]
            end
        end
        
        u0 = ones(N)
        tspan = (0.0, 0.2)
        jp = tridiagonal_sparsity(N)
        f = ODEFunction(large_sparse!, jac_prototype=jp)
        prob = ODEProblem(f, u0, tspan)
        
        sol_m, sol_s = compare_edge_cases(prob, "Large Sparse Pivot")
        @test sol_m.retcode == ReturnCode.Success
    end

    @testset "3. Intermittent Stiffness (Ignition)" begin
        # 1D system with a massive reaction spike.
        N = 50
        dx = 1.0/N
        function ignition!(du, u, p, t)
            # Reaction spike that is very hard for explicit solvers
            k = t < 0.05 ? 0.1 : 1e6
            for i in 1:N
                du[i] = -k * u[i]^2 + sin(i*t)
            end
        end
        u0 = ones(N)
        tspan = (0.0, 0.1)
        prob = ODEProblem(ignition!, u0, tspan)
        
        sol_m, sol_s = compare_edge_cases(prob, "Intermittent Stiffness")
        @test sol_m.retcode == ReturnCode.Success
    end

end
