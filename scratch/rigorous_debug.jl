using Frankenstein
using DifferentialEquations
using SparseArrays
using SparseDiffTools
using ADTypes

println("--- Rigorous Sparsity Debugging ---")

N = 100 
function sparse_f!(du, u, p, t)
    for i in 1:N
        du[i] = -u[i]
        if i > 1
            du[i] += 0.1 * u[i-1]
        end
        if i < N
            du[i] += 0.1 * u[i+1]
        end
    end
end
u0 = ones(N)
tspan = (0.0, 1.0)
prob = ODEProblem(sparse_f!, u0, tspan)

println("1. Initial Analysis...")
analysis = Frankenstein.Analysis.analyze_system_structure(prob)
println("Detected sparsity: ", analysis.is_sparse)
pattern_analysis = analysis.sparsity_pattern
println("Analysis Pattern nnz: ", pattern_analysis !== nothing ? nnz(pattern_analysis) : "nothing")

println("2. Comparing with SparseDiffTools.jacobian_sparsity directly...")
ad_func = (u) -> begin
    du = similar(u)
    sparse_f!(du, u, prob.p, prob.tspan[1])
    return du
end
pattern_direct = SparseDiffTools.jacobian_sparsity(ad_func, u0)
println("Direct SDT Pattern nnz: ", nnz(pattern_direct))

if pattern_analysis !== nothing && pattern_direct !== nothing
    mismatch = pattern_analysis != pattern_direct
    println("Patterns Match? ", !mismatch)
    if mismatch
        println("Diff count: ", sum(pattern_analysis .!= pattern_direct))
        # Check if one is a subset of the other
        println("Analysis subset of Direct? ", all(pattern_analysis .<= pattern_direct))
        println("Direct subset of Analysis? ", all(pattern_direct .<= pattern_analysis))
    end
end

println("3. Inspecting ODEFunction in Frankenstein.MonsterSolver...")
# Simulate what MonsterSolver does
raw_f = prob.f isa SciMLBase.ODEFunction ? prob.f.f : prob.f
new_f = SciMLBase.ODEFunction(raw_f; 
                   jac_prototype = (analysis.is_sparse && analysis.sparsity_pattern !== nothing) ? 
                                   analysis.sparsity_pattern : nothing)

println("new_f.jac_prototype nnz: ", new_f.jac_prototype !== nothing ? nnz(new_f.jac_prototype) : "nothing")

println("4. Attempting to build a ForwardDiffColorJacobianCache manually...")
try
    # Using the same logic as SciML/OrdinaryDiffEq would use
    if new_f.jac_prototype !== nothing
        # We need a colorvec
        using SparseMatrixColorings
        colorvec = SparseDiffTools.matrix_colors(new_f.jac_prototype)
        println("Colorvec size: ", length(colorvec))
        
        # Build cache
        # This is where the mismatch usually manifests if we use the wrong pattern
        jac_cache = ForwardDiffColorJacobianCache(ad_func, u0; 
                                                 colorvec=colorvec, 
                                                 sparsity=new_f.jac_prototype)
        println("Cache built successfully!")
        
        # Test the update
        A = copy(new_f.jac_prototype)
        ForwardDiffColorJacobianCache(ad_func, u0; colorvec=colorvec, sparsity=new_f.jac_prototype) # Re-build to be sure
        
        # The solver uses forwarddiff_color_jacobian!(A, ad_func, u, jac_cache)
        # We simulate the mismatch:
        println("Testing update on A...")
        SparseDiffTools.forwarddiff_color_jacobian!(A, ad_func, u0, jac_cache)
        println("Update successful!")
    end
catch e
    println("CAUGHT ERROR:")
    showerror(stdout, e)
    println()
end
