using Pkg
packages = [
    "DifferentialEquations", 
    "SciMLBase", 
    "OrdinaryDiffEq", 
    "Sundials", 
    "ForwardDiff", 
    "Enzyme", 
    "FiniteDiff", 
    "SparseDiffTools", 
    "Symbolics", 
    "LinearSolve", 
    "SparseArrays", 
    "LinearAlgebra", 
    "ADTypes", 
    "ModelingToolkit", 
    "DataStructures", 
    "Statistics",
    "Preferences"
]
Pkg.activate(".")
Pkg.add(packages)
