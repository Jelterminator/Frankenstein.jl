using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("test/runtests.jl")
