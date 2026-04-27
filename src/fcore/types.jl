# types.jl - Fundamental enums and structures for Frankenstein.jl

#==============================================================================#
# Enums for Problem Classification
#==============================================================================#

@enum SolverCategory begin
    EXPLICIT            # Pure explicit methods (Tsit5, DP5)
    STABILIZED_EXPLICIT # Explicit methods with large stability regions (ROCK, ESERK)
    STIFF               # Implicit/semi-implicit for STIFF problems (Rodas, QNDF)
    COMPOSITE           # IMEX and hybrid methods
    MULTISCALE          # Methods for multiple timescales
    SPARSE              # Sparse-matrix optimized methods
    ADAPTIVE            # Methods with advanced adaptivity
    PARALLEL            # Parallel/distributed methods
    SPECIALTY           # Special purpose (high precision, low memory, etc.)
end

@enum StiffnessLevel begin
    SL_NON_STIFF      # stiffness_ratio < 100 (Explicit territory)
    SL_MILDLY_STIFF   # 100 ≤ stiffness_ratio < 1000 (Explicit struggling)
    SL_STIFF          # 1000 ≤ stiffness_ratio < 10000 (Implicit territory)
    SL_VERY_STIFF     # 10000 ≤ stiffness_ratio < 100000 (BDF/IMEX territory)
    SL_EXTREMELY_STIFF # stiffness_ratio ≥ 100000 (Highly singular)
end

@enum SystemSize begin
    SS_SMALL_SYSTEM   # system_size < 50 (Dense, StaticArrays territory)
    SS_MEDIUM_SYSTEM  # 50 ≤ system_size < 500 (Dense BLAS territory)
    SS_LARGE_SYSTEM   # 500 ≤ system_size < 5000 (Sparse direct territory)
    SS_EXTREME_SYSTEM # system_size ≥ 5000 (Krylov/Parallel territory)
end

@enum AccuracyLevel begin  
    LOW_ACCURACY      # rtol > 1e-3
    STANDARD_ACCURACY # 1e-7 ≤ rtol ≤ 1e-3
    HIGH_ACCURACY     # rtol < 1e-7
end

#==============================================================================#
# Core Frankenstein Types
#==============================================================================#

"""
    SystemAnalysis{T}
"""
mutable struct SystemAnalysis{T}
    stiffness_ratio::T
    is_stiff::Bool
    sparsity_pattern::Any
    timescales::Vector{T}
    coupling_strength::T
    condition_number::T
    system_size::Int
    is_sparse::Bool
    jacobian::Any
    stable_count::Int
    last_update_step::Int
    last_update_t::T
    current_step::Int
    last_norm_du::T
    last_jacobian_update::Int
    last_reject_count::Int
    last_surgery_step::Int
    diagnostic_cooldown::Int
    watchdog_interval::Int
    current_category::SolverCategory
    history::Dict{Symbol, Any}
end

function SystemAnalysis{T}() where T
    return SystemAnalysis{T}(
        T(NaN), false, nothing, T[], T(NaN), T(NaN), 
        0, false, nothing, 0, -100, T(0), 0, T(0), 0, 0, -100, 50, 10000, EXPLICIT, Dict{Symbol, Any}()
    )
end

function SystemAnalysis()
    return SystemAnalysis{Float64}()
end

"""
    StepInfo{T, P}
"""
struct StepInfo{T, P}
    u::Vector{T}
    du::Vector{T}
    dt::T
    dt_prev::T
    rejects::Int
    nsteps::Int
    t::T
    p::P
    prob::SciMLBase.ODEProblem
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
                                   description::String="",
                                   references::Vector{String}=String[]) where {Alg}
        
        new{Alg}(alg, priority, category, min_accuracy, max_accuracy, 
                memory_efficiency, computational_cost, stability_score,
                stiffness_range, system_size_range, handles_sparse, 
                handles_mass_matrix, supports_events, description, references)
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
    
    if isnan(ratio) || ratio < 100
        return SL_NON_STIFF
    elseif ratio < 1000
        return SL_MILDLY_STIFF
    elseif ratio < 10000
        return SL_STIFF  
    elseif ratio < 100000
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
    
    if size < 50
        return SS_SMALL_SYSTEM
    elseif size < 500
        return SS_MEDIUM_SYSTEM
    elseif size < 5000
        return SS_LARGE_SYSTEM
    else
        return SS_EXTREME_SYSTEM
    end
end

"""
    classify_accuracy_level(rtol::Float64) -> AccuracyLevel

Classify the desired accuracy level based on relative tolerance.
"""
function classify_accuracy_level(rtol::Float64)
    if rtol > 1e-3
        return LOW_ACCURACY
    elseif rtol >= 1e-7
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
    return max_separation > 1000  # 3 orders of magnitude for "true" multiscale
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
    
    # 1. Trait-Based Bonuses (The "ASP" Mapping)
    
    # Multiscale & Timescale Bonus
    if has_multiscale_behavior(analysis)
        if rec.category == MULTISCALE
            priority += 2.5 # Huge boost for specialized multiscale solvers
        elseif rec.category == COMPOSITE
            priority += 1.0 # Moderate boost for IMEX/Composite
        end
    end
    
    # Coupling Sensitivity (Rice's ASP: Feature Space Mapping)
    # High coupling (dense interdependence) makes splitting methods inefficient or unstable.
    if analysis.coupling_strength > 0.7
        if rec.category in [COMPOSITE, MULTISCALE]
            priority -= (analysis.coupling_strength * 2.0) # Penalty scales with coupling
        end
    elseif analysis.coupling_strength < 0.2
        if rec.category in [COMPOSITE, MULTISCALE]
            priority += 1.0 # Boost splitting for loose coupling
        end
    end
    
    # Advanced Conditioning Check
    if !is_well_conditioned(analysis)
        # Logarithmic scaling for ill-conditioning bonus, capped to avoid infinite boosts
        cond_bonus = min(log10(max(1.0, analysis.condition_number)) / 4.0, 1.5)
        priority += (rec.stability_score * cond_bonus * 2.0)
    end

    # Penalty for low-order algorithms (ImplicitEuler) on large systems
    if (string(rec.algorithm) == "ImplicitEuler") && classify_system_size(analysis) >= SS_MEDIUM_SYSTEM
        priority -= 5.0 # Strongly discourage ImplicitEuler for large PDEs
    end

    # 2. User Preference Adjustments
    
    # Memory preference adjustment
    if prefer_memory
        priority += rec.memory_efficiency * 1.5
    end
    
    # Stability preference adjustment  
    if prefer_stability
        # If the user asks for stability AND the problem is stiff, double the weight
        stiff_mult = classify_stiffness(analysis) >= SL_STIFF ? 2.0 : 1.0
        priority += rec.stability_score * 1.5 * stiff_mult
    end
    
    # 3. Hardware & Scale Penalties (Amortized Cost Analysis)
    
    # Penalize high computational cost for large systems
    sys_size = classify_system_size(analysis)
    if sys_size == SS_LARGE_SYSTEM
        # Rice's ASP: Performance Space mapping. 
        # For large systems, we tolerate high cost ONLY if the problem is extremely stiff.
        stiffness = classify_stiffness(analysis)
        penalty_scale = if stiffness >= SL_EXTREMELY_STIFF 0.2
                        elseif stiffness >= SL_VERY_STIFF 0.5
                        else 1.5 end
        priority -= rec.computational_cost * penalty_scale
    elseif sys_size == SS_MEDIUM_SYSTEM
        priority -= rec.computational_cost * 0.4
    end
    
    # Boost sparse-aware algorithms for sparse systems
    if requires_sparse_handling(analysis) && rec.handles_sparse
        priority += 2.0 # Significant boost for sparse handling
    end
    
    return priority
end
