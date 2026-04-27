using Frankenstein
using ADTypes
using SparseDiffTools

println("Checking for AutoSparseForwardDiff...")
try
    println("Type of AutoSparseForwardDiff: ", typeof(Frankenstein.Backends.AutoSparseForwardDiff))
catch e
    println("Caught error: ", e)
end

println("Checking for ADTypes.AutoSparse...")
try
    println("AutoSparse exists.")
catch e
    println("AutoSparse does not exist: ", e)
end
