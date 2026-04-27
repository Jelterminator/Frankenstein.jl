using Frankenstein
using DifferentialEquations
using SparseArrays

# 2D Wave
N = 25
L = sparse(I, N*N, N*N) # dummy
jp = [spzeros(N*N, N*N) L; L spzeros(N*N, N*N)] + sparse(I, 2*N*N, 2*N*N)
f = ODEFunction((du,u,p,t)->nothing, jac_prototype=jp)
prob = ODEProblem(f, rand(2*N*N), (0.0, 1.0))

analysis = Frankenstein.Analysis.analyze_system_structure(prob)
recs = Frankenstein.Solvers.get_all_recommendations(analysis)

println("--- 2D WAVE RECS ---")
for r in recs[1:5]
    pri = Frankenstein.FCore.compute_adjusted_priority(r, analysis, prefer_stability=true)
    println(r.algorithm, " | Base: ", r.priority, " | Adj: ", pri)
end
