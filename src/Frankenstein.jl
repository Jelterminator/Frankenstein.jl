module Frankenstein

using DifferentialEquations
using SciMLBase
using ModelingToolkit
using ForwardDiff

# Include submodules
include("fcore/fcore.jl")
include("utilities/utilities.jl")
include("backends/backends.jl")
include("solvers/solvers.jl")
include("analysis/analysis.jl")
include("monitoring/monitoring.jl")
include("adaptation/adaptation.jl")
include("preconditioning/preconditioning.jl")
include("splitting/splitting.jl")
include("MonsterSolver.jl")

# Using submodules
using .FCore, .Backends, .Solvers, .Analysis, .MonsterSolver
using .Adaptation, .Preconditioning, .Splitting, .Monitoring, .Utilities

export FrankensteinSolver, SolverConfiguration, Monster, solve, analyze_system

# Convenience constructor to match the README
# We use 'Monster' instead of 'Frankenstein' to avoid conflict with the module name.
Monster() = FrankensteinSolver()

# SciML Integration dispatches
function SciMLBase.solve(prob::SciMLBase.ODEProblem, Fs::FrankensteinSolver; kwargs...)
    return monster_solve!(prob, Fs; kwargs...)
end

function SciMLBase.__solve(prob::SciMLBase.ODEProblem, Fs::FrankensteinSolver; kwargs...)
    return monster_solve!(prob, Fs; kwargs...)
end

end
