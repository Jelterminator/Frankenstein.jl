using Frankenstein
using SciMLBase, LinearAlgebra, SparseArrays, SparseMatrixColorings, ADTypes

function laplacian_2d_sparsity(N)
    I_idx, J_idx = Int[], Int[]
    for j in 1:N, i in 1:N
        idx = (j-1)*N + i
        push!(I_idx, idx); push!(J_idx, idx)
        if i > 1; push!(I_idx, idx); push!(J_idx, idx-1); end
        if i < N; push!(I_idx, idx); push!(J_idx, idx+1); end
        if j > 1; push!(I_idx, idx); push!(J_idx, idx-N); end
        if j < N; push!(I_idx, idx); push!(J_idx, idx+N); end
    end
    S = sparse(I_idx, J_idx, ones(length(I_idx)))
    dropzeros!(S)
    return S
end

function run_gauntlet()
    println("=== FRANKENSTEIN INTELLIGENCE GAUNTLET ===")
    flush(stdout)

    # --- Benchmark 1: Oregonator ---
    println("[Benchmark 1] Oregonator (Small & Stiff)")
    function oregonator!(du, u, p, t)
        s, w, q = 77.27, 0.161, 8.375e-6
        du[1] = s*(u[2] + u[1]*(1.0 - q*u[1] - u[2]))
        du[2] = (u[3] - (1.0 + u[1])*u[2]) / s
        du[3] = w*(u[1] - u[3])
    end
    u0_oregon = Float64[1.0, 2.0, 3.0]
    p_oregon = Float64[]
    tspan_oregon = (0.0, 2.0)
    prob_oregon = ODEProblem(oregonator!, u0_oregon, tspan_oregon, p_oregon)
    sol_oregon = solve(prob_oregon, Monster())
    println("  Success: sol_oregon retcode = $(sol_oregon.retcode)")

    # --- Benchmark 2: Kuramoto ---
    println("[Benchmark 2] Kuramoto (100% Dense, n=100)")
    let N=100, K=5.0
        ω = rand(N)
        function kuramoto!(du, u, p, t)
            for i in 1:N
                sum_terms = 0.0
                for j in 1:N; sum_terms += sin(u[j] - u[i]); end
                du[i] = ω[i] + (K/N) * sum_terms
            end
        end
        prob_kura = ODEProblem(kuramoto!, rand(N), (0.0, 5.0))
        sol_kura = solve(prob_kura, Monster())
        println("  Success: sol_kura retcode = $(sol_kura.retcode)")
    end

    # --- Benchmark 3: 2D Heat ---
    println("[Benchmark 3] 2D Heat (Sparse, n=900)")
    let N=30, dx=1.0/(30+1), c=0.1/dx^2
        function heat!(du,u,p,t)
            for j in 1:N, i in 1:N
                idx=Int((j-1)*N+i)
                l = i > 1 ? u[idx-1] : 0.0; r = i < N ? u[idx+1] : 0.0
                d = j > 1 ? u[idx-N] : 0.0; u_p = j < N ? u[idx+N] : 0.0
                du[idx]=c*(l+r+d+u_p-4*u[idx])
            end
        end
        ff = ODEFunction(heat!, jac_prototype=laplacian_2d_sparsity(N))
        prob_heat = ODEProblem(ff, rand(N^2), (0.0, 0.1))
        sol_heat = solve(prob_heat, Monster())
        println("  Success: sol_heat retcode = $(sol_heat.retcode)")
    end

    println("=== GAUNTLET COMPLETE: FRANKENSTEIN IS ASCENDED ===")
end

run_gauntlet()
