using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("test/runtests.jl")
include("test/test_pde_suite.jl")
include("test/edge_cases.jl")
