using Frankenstein
using SciMLBase, SparseArrays, LinearAlgebra

function eye_exam()
    println("--- Frankenstein Eye Exam ---")
    N = 10
    jp = sparse(I, N^2, N^2) # Simple diagonal prototype
    
    f!(du, u, p, t) = du .= u
    ff = ODEFunction(f!, jac_prototype=jp)
    prob = ODEProblem(ff, rand(N^2), (0.0, 1.0))
    
    println("Checking ODEFunction for jac_prototype...")
    println("  propertynames(ff): ", propertynames(ff))
    
    # Run the actual Frankenstein detector
    try
        # Attempt to access the internal detector
        detector = Frankenstein.Sparsity.detect_sparsity_patterns
        result = detector(prob)
        
        if result === nothing
            println("RESULT: ❌ DETECTOR FOUND NOTHING")
        else
            println("RESULT: ✅ DETECTOR FOUND SPARSITY: ", typeof(result), " (nnz=$(nnz(result)))")
        end
    catch e
        println("ERROR CALLING DETECTOR: ", e)
    end
end

eye_exam()
