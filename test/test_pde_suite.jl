using Frankenstein
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Test

# Helper to create 2D Laplacian sparsity pattern
function laplacian_2d_sparsity(N)
    I_idx = Int[]
    J_idx = Int[]
    for j in 1:N, i in 1:N
        idx = (j-1)*N + i
        # Self
        push!(I_idx, idx); push!(J_idx, idx)
        # Neighbors
        if i > 1;  push!(I_idx, idx); push!(J_idx, idx-1); end
        if i < N;  push!(I_idx, idx); push!(J_idx, idx+1); end
        if j > 1;  push!(I_idx, idx); push!(J_idx, idx-N); end
        if j < N;  push!(I_idx, idx); push!(J_idx, idx+N); end
    end
    V_idx = ones(Float64, length(I_idx))
    return sparse(I_idx, J_idx, V_idx)
end

# Benchmarking Helper
function compare_solvers(prob, name)
    @info "--- Benchmarking: $name ---"
    
    # 1. SciML PolyAlgorithm (Default)
    t_native = @elapsed sol_native = DifferentialEquations.solve(prob)
    
    # 2. Frankenstein Monster
    t_monster = @elapsed sol_monster = Frankenstein.solve(prob, Monster())
    
    println("  [Native] Time: $(round(t_native, digits=4))s | Steps: $(length(sol_native.t)) | Retcode: $(sol_native.retcode)")
    println("  [Monster] Time: $(round(t_monster, digits=4))s | Steps: $(length(sol_monster.t)) | Retcode: $(sol_monster.retcode)")
    
    if t_monster < t_native
        println("  >> Monster is $(round(t_native/t_monster, digits=2))x FASTER")
    else
        println("  >> Native is $(round(t_monster/t_native, digits=2))x FASTER")
    end
    println("------------------------------------------")
    
    return sol_monster, sol_native
end

# 2D Heat Equation (diffusion)
@testset "2D Heat Equation" begin
    N = 50
    dx = 1.0 / (N + 1)
    D = 0.1
    c = D / dx^2
    function heat2d!(du, u, p, t)
        for j in 1:N, i in 1:N
            idx = (j-1)*N + i
            left  = i > 1   ? u[idx-1] : 0.0
            right = i < N   ? u[idx+1] : 0.0
            down  = j > 1   ? u[idx-N] : 0.0
            up    = j < N   ? u[idx+N] : 0.0
            center = u[idx]
            du[idx] = c * (left + right + down + up - 4*center)
        end
    end
    u0 = zeros(N*N)
    u0[div(N, 2) + N*div(N, 2)] = 1.0
    
    # Provide sparsity pattern
    jp = laplacian_2d_sparsity(N)
    f = ODEFunction(heat2d!, jac_prototype=jp)
    prob = ODEProblem(f, u0, (0.0, 1.0)) 
    
    sol, _ = compare_solvers(prob, "2D Heat (100% Diffusion)")
    
    @test sol.retcode == ReturnCode.Success
    @test all(sol.u[end] .>= -1e-10)
    @test sum(sol.u[end]) <= sum(u0) + 1e-10
end

# 2D Wave Equation
@testset "2D Wave Equation" begin
    N = 25
    dx = 1.0 / (N + 1)
    c_wave = 1.0
    function wave2d!(du, u, p, t)
        v = view(u, 1:N*N)
        w = view(u, N*N+1:2*N*N)
        dv = view(du, 1:N*N)
        dw = view(du, N*N+1:2*N*N)
        dv .= w
        for j in 1:N, i in 1:N
            idx = (j-1)*N + i
            left  = i > 1   ? v[idx-1] : 0.0
            right = i < N   ? v[idx+1] : 0.0
            down  = j > 1   ? v[idx-N] : 0.0
            up    = j < N   ? v[idx+N] : 0.0
            center = v[idx]
            dw[idx] = c_wave^2 * (left + right + down + up - 4*center) / dx^2
        end
    end
    v0 = zeros(N*N)
    w0 = zeros(N*N)
    for j in 1:N, i in 1:N
        x = i*dx; y = j*dx
        v0[(j-1)*N + i] = exp(-100*((x-0.5)^2 + (y-0.5)^2))
    end
    u0 = [v0; w0]
    
    # 2D Wave sparsity (block structured)
    L = laplacian_2d_sparsity(N)
    Z = spzeros(N*N, N*N)
    Id = sparse(I, N*N, N*N)
    jp = [Z Id; L Z]
    
    f = ODEFunction(wave2d!, jac_prototype=jp)
    prob = ODEProblem(f, u0, (0.0, 0.5))
    
    sol, _ = compare_solvers(prob, "2D Wave Equation")
    @test sol.retcode == ReturnCode.Success
end

# Burgers' Equation
@testset "Burgers' Equation" begin
    N = 30
    dx = 1.0 / (N + 1)
    ν = 0.01
    function burgers!(du, u, p, t)
        for j in 1:N, i in 1:N
            idx = (j-1)*N + i
            uij = u[idx]
            left  = i > 1   ? u[idx-1] : 0.0
            right = i < N   ? u[idx+1] : 0.0
            down  = j > 1   ? u[idx-N] : 0.0
            up    = j < N   ? u[idx+N] : 0.0
            
            diff = ν * ( (left - 2*uij + right) + (down - 2*uij + up) ) / dx^2
            conv = -uij * ( (right - left) / (2dx) )
            du[idx] = diff + conv
        end
    end
    u0 = [sin(pi * (i*dx)) * sin(pi * (j*dx)) for j in 1:N, i in 1:N]
    u0 = reshape(u0, N*N)
    
    jp = laplacian_2d_sparsity(N) # Same sparsity as Heat for this stencil
    f = ODEFunction(burgers!, jac_prototype=jp)
    prob = ODEProblem(f, u0, (0.0, 0.5))
    
    sol, _ = compare_solvers(prob, "Burgers' Equation")
    @test sol.retcode == ReturnCode.Success
end

# Advection‑Diffusion‑Reaction
@testset "Advection‑Diffusion‑Reaction" begin
    N = 25
    dx = 1.0 / (N + 1)
    D_adr = 0.01
    vx = 0.5; vy = 0.5
    k_adr = 50.0 
    function adr!(du, u, p, t)
        for j in 1:N, i in 1:N
            idx = (j-1)*N + i
            uij = u[idx]
            left  = i > 1   ? u[idx-1] : 0.0
            right = i < N   ? u[idx+1] : 0.0
            down  = j > 1   ? u[idx-N] : 0.0
            up    = j < N   ? u[idx+N] : 0.0
            
            diffusion = D_adr * ( (left - 2*uij + right) + (down - 2*uij + up) ) / dx^2
            advection = -vx * (right - left) / (2dx) - vy * (up - down) / (2dx)
            reaction = k_adr * uij * (1.0 - uij)
            du[idx] = diffusion + advection + reaction
        end
    end
    u0 = zeros(N*N)
    u0[div(N, 2) + N*div(N, 2)] = 0.5
    
    jp = laplacian_2d_sparsity(N)
    f = ODEFunction(adr!, jac_prototype=jp)
    prob = ODEProblem(f, u0, (0.0, 0.1))
    
    sol, _ = compare_solvers(prob, "ADR Equation")
    @test sol.retcode == ReturnCode.Success
end

# Gray‑Scott Reaction‑Diffusion
@testset "Gray‑Scott Reaction‑Diffusion" begin
    N = 30
    dx = 1.0 / (N + 1)
    Du = 0.16; Dv = 0.08
    F_gs = 0.035; k_gs = 0.06
    
    function gray_scott!(du, u_full, p, t)
        u_v = view(u_full, 1:N*N)
        v_v = view(u_full, N*N+1:2*N*N)
        du_u = view(du, 1:N*N)
        du_v = view(du, N*N+1:2*N*N)
        
        for j in 1:N, i in 1:N
            idx = (j-1)*N + i
            uij = u_v[idx]
            vij = v_v[idx]
            
            l_u = i > 1 ? u_v[idx-1] : uij
            r_u = i < N ? u_v[idx+1] : uij
            d_u = j > 1 ? u_v[idx-N] : uij
            u_u = j < N ? u_v[idx+N] : uij
            
            l_v = i > 1 ? v_v[idx-1] : vij
            r_v = i < N ? v_v[idx+1] : vij
            d_v = j > 1 ? v_v[idx-N] : vij
            u_v_cell = j < N ? v_v[idx+N] : vij
            
            lap_u = (l_u + r_u + d_u + u_u - 4*uij) / dx^2
            lap_v = (l_v + r_v + d_v + u_v_cell - 4*vij) / dx^2
            
            uv2 = uij * vij^2
            du_u[idx] = Du * lap_u - uv2 + F_gs * (1.0 - uij)
            du_v[idx] = Dv * lap_v + uv2 - (F_gs + k_gs) * vij
        end
    end
    
    u0_gs = ones(N*N)
    v0_gs = zeros(N*N)
    mid = div(N, 2)
    for j in (mid-2):(mid+2), i in (mid-2):(mid+2)
        v0_gs[(j-1)*N + i] = 0.25
        u0_gs[(j-1)*N + i] = 0.5
    end
    u0_full = [u0_gs; v0_gs]
    
    L = laplacian_2d_sparsity(N)
    jp = [L L; L L] # Conservative upper bound on sparsity
    f = ODEFunction(gray_scott!, jac_prototype=jp)
    prob = ODEProblem(f, u0_full, (0.0, 5.0))
    
    sol, _ = compare_solvers(prob, "Gray-Scott Equation")
    @test sol.retcode == ReturnCode.Success
end
