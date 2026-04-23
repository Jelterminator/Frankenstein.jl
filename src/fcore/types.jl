# types.jl - Fundamental enums and structures for Frankenstein.jl

#==============================================================================#
# Enums for Problem Classification
#==============================================================================#

@enum SolverCategory begin
    EXPLICIT        # Pure explicit methods
    STIFF           # Implicit/semi-implicit for STIFF problems  
    COMPOSITE      # IMEX and hybrid methods
    MULTISCALE     # Methods for multiple timescales
    SPARSE         # Sparse-matrix optimized methods
    ADAPTIVE       # Methods with advanced adaptivity
    PARALLEL       # Parallel/distributed methods
    SPECIALTY      # Special purpose (high precision, low memory, etc.)
end

@enum StiffnessLevel begin
    SL_NON_STIFF      # stiffness_ratio < 10
    SL_MILDLY_STIFF   # 10 ≤ stiffness_ratio < 100  
    SL_STIFF          # 100 ≤ stiffness_ratio < 1000
    SL_VERY_STIFF     # 1000 ≤ stiffness_ratio < 10000
    SL_EXTREMELY_STIFF # stiffness_ratio ≥ 10000
end

@enum SystemSize begin
    SS_SMALL_SYSTEM   # system_size < 100
    SS_MEDIUM_SYSTEM  # 100 ≤ system_size < 1000
    SS_LARGE_SYSTEM   # system_size ≥ 1000
end

@enum AccuracyLevel begin  
    LOW_ACCURACY    # rtol > 1e-3
    STANDARD_ACCURACY # 1e-6 ≤ rtol ≤ 1e-3
    HIGH_ACCURACY   # rtol < 1e-6
end

#==============================================================================#
# Algorithm Recommendation Structure
#==============================================================================#

"""
    AlgorithmRecommendation{Alg}

Represents a single algorithm recommendation with metadata about its suitability
for different problem characteristics.
"""
struct AlgorithmRecommendation{Alg}
    algorithm::Alg                 # Now stores the constructor (e.g. Rodas5P) or a factory function
    priority::Float64              # Base priority score (0-10)
    category::SolverCategory       # Primary category this algorithm belongs to
    
    # Performance characteristics
    min_accuracy::Float64          # Minimum rtol this works well for
    max_accuracy::Float64          # Maximum rtol this works well for  
    memory_efficiency::Float64     # 0-1 scale, 1 = very memory efficient
    computational_cost::Float64    # 0-1 scale, 1 = very expensive per step
    stability_score::Float64       # 0-1 scale, 1 = extremely stable
    
    # Applicability conditions
    stiffness_range::Tuple{StiffnessLevel, StiffnessLevel}  # (min, max) stiffness levels
    system_size_range::Tuple{SystemSize, SystemSize}       # (min, max) system sizes
    handles_sparse::Bool           # Can efficiently handle sparse systems
    handles_mass_matrix::Bool      # Can handle mass matrices
    supports_events::Bool          # Supports event handling
    is_sundials::Bool              # Flag for Sundials-specific keyword handling
    
    # Metadata
    description::String
    references::Vector{String}     # Literature references
    
    # Constructor with sensible defaults
    function AlgorithmRecommendation(alg::Alg, priority::Float64, category::SolverCategory;
                                   min_accuracy::Float64=1e-12,
                                   max_accuracy::Float64=1e-2,
                                   memory_efficiency::Float64=0.8,
                                   computational_cost::Float64=0.5,
                                   stability_score::Float64=0.8,
                                   stiffness_range::Tuple{StiffnessLevel, StiffnessLevel}=(SL_NON_STIFF, SL_EXTREMELY_STIFF),
                                   system_size_range::Tuple{SystemSize, SystemSize}=(SS_SMALL_SYSTEM, SS_LARGE_SYSTEM),
                                   handles_sparse::Bool=false,
                                   handles_mass_matrix::Bool=false,
                                   supports_events::Bool=true,
                                   is_sundials::Bool=false,
                                   description::String="",
                                   references::Vector{String}=String[]) where {Alg}
        
        new{Alg}(alg, priority, category, min_accuracy, max_accuracy, 
                memory_efficiency, computational_cost, stability_score,
                stiffness_range, system_size_range, handles_sparse, 
                handles_mass_matrix, supports_events, is_sundials, description, references)
    end
end


#==============================================================================#
# Problem Characterization Functions
#==============================================================================#

"""
    classify_stiffness(analysis::SystemAnalysis) -> StiffnessLevel

Classify the stiffness level based on the stiffness ratio.
"""
function classify_stiffness(analysis::SystemAnalysis)
    ratio = analysis.stiffness_ratio
    
    if isnan(ratio) || ratio < 10
        return SL_NON_STIFF
    elseif ratio < 100
        return SL_MILDLY_STIFF
    elseif ratio < 1000
        return SL_STIFF  
    elseif ratio < 10000
        return SL_VERY_STIFF
    else
        return SL_EXTREMELY_STIFF
    end
end

"""
    classify_system_size(analysis::SystemAnalysis) -> SystemSize

Classify the system size.
"""
function classify_system_size(analysis::SystemAnalysis)
    size = analysis.system_size
    
    if size < 100
        return SS_SMALL_SYSTEM
    elseif size < 1000
        return SS_MEDIUM_SYSTEM
    else
        return SS_LARGE_SYSTEM
    end
end

"""
    classify_accuracy_level(rtol::Float64) -> AccuracyLevel

Classify the desired accuracy level based on relative tolerance.
"""
function classify_accuracy_level(rtol::Float64)
    if rtol > 1e-3
        return LOW_ACCURACY
    elseif rtol >= 1e-6
        return STANDARD_ACCURACY
    else
        return HIGH_ACCURACY
    end
end

"""
    has_multiscale_behavior(analysis::SystemAnalysis) -> Bool

Determine if the system exhibits significant multiscale behavior.
"""
function has_multiscale_behavior(analysis::SystemAnalysis)
    timescales = analysis.timescales
    
    if length(timescales) < 2
        return false
    end
    
    # Check for significant separation in timescales
    sorted_scales = sort(abs.(timescales[timescales .!= 0]))
    
    if length(sorted_scales) < 2
        return false
    end
    
    max_separation = sorted_scales[end] / sorted_scales[1]
    return max_separation > 100  # More than 2 orders of magnitude
end

"""
    is_well_conditioned(analysis::SystemAnalysis) -> Bool

Check if the system is well-conditioned.
"""  
function is_well_conditioned(analysis::SystemAnalysis)
    cond_num = analysis.condition_number
    return !isnan(cond_num) && cond_num < 1000
end

"""
    requires_sparse_handling(analysis::SystemAnalysis) -> Bool

Determine if sparse matrix handling would be beneficial.
"""
function requires_sparse_handling(analysis::SystemAnalysis)
    return analysis.is_sparse && analysis.system_size > 50
end

#==============================================================================#
# Scoring and Filtering Utilities  
#==============================================================================#

"""
    is_applicable(rec::AlgorithmRecommendation, analysis::SystemAnalysis, rtol::Float64) -> Bool

Check if an algorithm recommendation is applicable to the given problem.
"""
function is_applicable(rec::AlgorithmRecommendation, analysis::SystemAnalysis, rtol::Float64)
    # Check accuracy requirements
    if !(rec.min_accuracy <= rtol <= rec.max_accuracy)
        return false
    end
    
    # Check stiffness compatibility
    stiffness = classify_stiffness(analysis)
    stiff_min, stiff_max = rec.stiffness_range
    if !(stiff_min <= stiffness <= stiff_max)
        return false
    end
    
    # Check system size compatibility
    sys_size = classify_system_size(analysis)
    size_min, size_max = rec.system_size_range
    if !(size_min <= sys_size <= size_max)
        return false
    end
    
    # Check sparse requirements
    if requires_sparse_handling(analysis) && !rec.handles_sparse
        return false
    end

    # Explicitly prevent sparse-optimized algorithms (categories like SPARSE)
    # for non-sparse problems to avoid MethodErrors in linear solvers.
    if rec.category == SPARSE && !analysis.is_sparse
        return false
    end
    
    return true
end

"""
    compute_adjusted_priority(rec::AlgorithmRecommendation, analysis::SystemAnalysis; 
                            prefer_memory::Bool=false, prefer_stability::Bool=false) -> Float64

Compute priority score adjusted for specific problem characteristics and preferences.
"""
function compute_adjusted_priority(rec::AlgorithmRecommendation, analysis::SystemAnalysis;
                                 prefer_memory::Bool=false, prefer_stability::Bool=false)
    
    priority = rec.priority
    
    # Memory preference adjustment
    if prefer_memory
        priority += rec.memory_efficiency * 1.5
    end
    
    # Stability preference adjustment  
    if prefer_stability || !is_well_conditioned(analysis)
        priority += rec.stability_score * 1.0
    end
    
    # Penalize high computational cost for large systems
    sys_size = classify_system_size(analysis)
    if sys_size == SS_LARGE_SYSTEM
        priority -= rec.computational_cost * 0.8
    elseif sys_size == SS_MEDIUM_SYSTEM
        priority -= rec.computational_cost * 0.4
    end
    
    # Boost sparse-aware algorithms for sparse systems
    if requires_sparse_handling(analysis) && rec.handles_sparse
        priority += 1.0
    end
    
    return priority
end
