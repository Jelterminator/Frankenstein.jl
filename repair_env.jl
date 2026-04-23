using Pkg
Pkg.activate(".")
println("Repairing environment...")

# Explicitly add missing dependencies that were causing extension clashes
Pkg.add(["Plots", "YAML", "IJulia"])

# Resolve and precompile all
Pkg.resolve()
Pkg.precompile()

println("Environment repaired and precompiled successfully!")
