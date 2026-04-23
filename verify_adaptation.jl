# verify_adaptation.jl
using Frankenstein
using SciMLBase
using OrdinaryDiffEq

# Define a simple non-stiff problem
f(u,p,t) = 1.01u
u0 = 1/2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

println("Analyzing system...")
sa = analyze_system(prob)

println("Creating adaptive composite solver...")
try
    comp = Frankenstein.Adaptation.create_adaptive_composite(sa)
    println("Successfully created CompositeAlgorithm!")
    println("Algorithms involved: ", [typeof(alg) for alg in comp.algs])
    
    # Minimal test: solve
    println("Running test solve...")
    sol = solve(prob, comp, reltol=1e-6, abstol=1e-6)
    println("Solve successful with ", length(sol.t), " steps.")
catch e
    @error "Adaptation setup failed" exception=(e, catch_backtrace())
    exit(1)
end
