using Frankenstein
using Test
using LinearAlgebra
using SparseArrays
using YAML
using ForwardDiff
using SciMLBase

# Include the chemistry logic
include(joinpath(@__DIR__, "combustion", "Chemistry.jl"))

@testset "Grand Combustion PDE Test" begin
    @info "Initializing Grand Combustion PDE Test (1D Premixed Flame)"
    
    # 1. Setup Mechanism and Parameters
    mech_path = joinpath(@__DIR__, "combustion", "LuMechanism.yaml")
    mechanism_data = load_mechanism_data(mech_path)
    species_list, reaction_list, species_index_map = process_chemical_data(mechanism_data)
    kinetics_data = build_kinetics_list(reaction_list)
    
    num_species = length(species_list)
    num_vars = num_species + 1 # T + species mass fractions
    
    # 2. Domain and Grid
    L = 0.02 # 2 cm
    nx = 30  # Slightly reduced for speed, still high fidelity (930 equations)
    dx = L / (nx - 1)
    x = collect(range(0, L, length=nx))
    
    # Pressure (Pa)
    P = 101325.0
    R_gas = 8.314
    
    # 3. Discretization Functions
    # Pre-build constant matrices
    S = build_stoichiometric_matrix(reaction_list, num_species)
    
    # Build Sparsity Prototype (Important for SPEED)
    # Each point connects to itself (reactions) and its 2 neighbors (transport)
    I_idx = Int[]
    J_idx = Int[]
    for j in 1:nx
        vars_j = ((j-1)*num_vars+1):(j*num_vars)
        # Self-connections (Dense block for reactions at each point)
        for v1 in vars_j, v2 in vars_j
            push!(I_idx, v1); push!(J_idx, v2)
        end
        # Neighbor connections (Diagonal blocks for transport)
        if j > 1
            vars_prev = ((j-2)*num_vars+1):((j-1)*num_vars)
            for i in 1:num_vars
                push!(I_idx, vars_j[i]); push!(J_idx, vars_prev[i])
            end
        end
        if j < nx
            vars_next = (j*num_vars+1):((j+1)*num_vars)
            for i in 1:num_vars
                push!(I_idx, vars_j[i]); push!(J_idx, vars_next[i])
            end
        end
    end
    jac_proto = sparse(I_idx, J_idx, ones(length(I_idx)))
    
    function combustion_pde!(du, u, p, t)
        u_mat = reshape(u, num_vars, nx)
        du_mat = reshape(du, num_vars, nx)
        fill!(du, 0.0)
        
        # Buffer for internal calculations
        r_buffer = zeros(eltype(u), length(reaction_list))
        
        for j in 2:(nx-1)
            T = u_mat[1, j]
            C = @view u_mat[2:end, j]
            
            # --- Reaction Source Term ---
            compute_reaction_rates!(r_buffer, @view(u_mat[:, j]), kinetics_data, species_list)
            reaction_source = S * r_buffer
            
            # --- Heat Release and Thermodynamics ---
            q_dot = 0.0
            total_cv = 0.0
            for i in 1:num_species
                h = h0(T, species_list[i].thermo) - R_gas * T
                cv = species_cp(T, species_list[i].thermo) - R_gas
                q_dot -= h * reaction_source[i]
                total_cv += C[i] * cv
            end
            total_cv = max(total_cv, 1.0) 
            
            # --- Transport ---
            dTdx2 = (u_mat[1, j+1] - 2*u_mat[1, j] + u_mat[1, j-1]) / dx^2
            kappa = conductivity(T)
            du_mat[1, j] = (kappa * dTdx2 + q_dot) / total_cv
            
            for i in 1:num_species
                D_i = 5e-5 
                dCdx2 = (u_mat[i+1, j+1] - 2*u_mat[i+1, j] + u_mat[i+1, j-1]) / dx^2
                du_mat[i+1, j] = D_i * dCdx2 + reaction_source[i]
            end
        end
        du_mat[:, nx] .= (u_mat[:, nx] .- u_mat[:, nx-1]) / dx 
    end
    
    # 4. Initial Conditions
    u0 = zeros(num_vars * nx)
    u0_mat = reshape(u0, num_vars, nx)
    
    cfg_inlet = ChemistryConfig(temperature=300.0, pressure=P, 
                                fuel_mixture=Dict("CH4"=>0.05, "H2"=>0.04), air_percentage=0.91)
    c_inlet = initialize_concentrations(cfg_inlet, species_list, species_index_map)
    for j in 1:nx; u0_mat[:, j] .= c_inlet; end
    
    spark_center = Int(nx ÷ 2)
    for j in (spark_center-2):(spark_center+2)
        u0_mat[1, j] = 2500.0
        for rad in ["OH", "O", "H", "HO2", "CH3", "HO2"]
            if haskey(species_index_map, rad)
                idx = species_index_map[rad]
                u0_mat[idx+1, j] += 0.1 
            end
        end
    end
    
    tspan = (0.0, 5e-4) 
    
    # Wrap in ODEFunction with jac_prototype
    ff = ODEFunction(combustion_pde!, jac_prototype=jac_proto)
    prob = ODEProblem(ff, u0, tspan)
    
    step_counter = Ref(0)
    cb = DiscreteCallback((u,t,integrator)->true, integrator -> begin
        step_counter[] += 1
        if step_counter[] % 10 == 0
            u_mat = reshape(integrator.u, num_vars, nx)
            @info "[Combustion Sim] t=$(round(integrator.t, sigdigits=4))s, dt=$(round(integrator.dt, sigdigits=4))s, Max T=$(round(maximum(u_mat[1, :]), digits=1))K"
        end
    end)
    
    @info "Starting Optimized Sparse Dual-Fuel Combustion PDE Test..."
    sol = Frankenstein.solve(prob, FrankensteinSolver(), reltol=1e-4, abstol=1e-6, callback=cb)

    
    @test sol.retcode == SciMLBase.ReturnCode.Success
    
    # 6. Check Results
    final_T_max = maximum(reshape(sol.u[end], num_vars, nx)[1, :])
    @info "Final Max Temperature: $final_T_max K"
    @test final_T_max > 1500.0 # Verify ignition occurred
    
    @info "Ultimate Combustion PDE Test Passed!"
end
