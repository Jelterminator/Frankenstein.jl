# algorithm_selector.jl - Algorithm selection and configuration

module AlgorithmSelector

using ADTypes
using ..FCore: SystemAnalysis, AbstractSolverStrategy, AlgorithmRecommendation, SolverCategory, StiffnessLevel, SystemSize, AccuracyLevel, is_applicable, compute_adjusted_priority, classify_stiffness, classify_system_size, classify_accuracy_level, requires_sparse_handling, is_well_conditioned, has_multiscale_behavior, SL_NON_STIFF, SL_MILDLY_STIFF, SL_STIFF, SL_VERY_STIFF, SL_EXTREMELY_STIFF, SS_SMALL_SYSTEM, SS_MEDIUM_SYSTEM, SS_LARGE_SYSTEM

using ..ExplicitSolvers: get_explicit_recommendations
using ..StiffSolvers: get_stiff_recommendations
using ..CompositeSolvers: get_composite_recommendations
using ..MultiscaleSolvers: get_multiscale_recommendations
using ..SparseSolvers: get_sparse_recommendations
using ..AdaptiveSolvers: get_adaptive_recommendations
using ..ParallelSolvers: get_parallel_recommendations
using ..SpecialtySolvers: get_specialty_recommendations

#==============================================================================#
# Unified Recommendation Interface
#==============================================================================#

"""
    get_all_recommendations(analysis::SystemAnalysis; kwargs...)

Gather recommendations from all solver strategy modules and return them
sorted by adjusted priority (highest first).
"""
function get_all_recommendations(analysis::SystemAnalysis; rtol::Float64=1e-6,
                                 prefer_memory::Bool=false,
                                 prefer_stability::Bool=true)
    # Collect from each strategy
    recs = vcat(
        get_explicit_recommendations(analysis; rtol=rtol,
                                     prefer_memory=prefer_memory,
                                     prefer_stability=prefer_stability),
        get_stiff_recommendations(analysis; rtol=rtol,
                                  prefer_memory=prefer_memory,
                                  prefer_stability=prefer_stability),
        get_composite_recommendations(analysis; rtol=rtol,
                                      prefer_memory=prefer_memory,
                                      prefer_stability=prefer_stability),
        get_multiscale_recommendations(analysis; rtol=rtol,
                                       prefer_memory=prefer_memory,
                                       prefer_stability=prefer_stability),
        get_sparse_recommendations(analysis; rtol=rtol,
                                   prefer_memory=prefer_memory,
                                   prefer_stability=prefer_stability),
        get_adaptive_recommendations(analysis; rtol=rtol,
                                     prefer_memory=prefer_memory,
                                     prefer_stability=prefer_stability),
        get_parallel_recommendations(analysis; rtol=rtol,
                                     prefer_memory=prefer_memory,
                                     prefer_stability=prefer_stability),
        get_specialty_recommendations(analysis; rtol=rtol,
                                      prefer_memory=prefer_memory,
                                      prefer_stability=prefer_stability)
    )
    # sort by adjusted priority
    return sort(recs; by = rec -> -compute_adjusted_priority(rec, analysis;
                                  prefer_memory=prefer_memory,
                                  prefer_stability=prefer_stability))
end

"""
    select_algorithm(analysis::SystemAnalysis; kwargs...)

Select the single best algorithm recommendation for the problem.
Returns the top AlgorithmRecommendation.
"""
function select_algorithm(analysis::SystemAnalysis; kwargs...)
    recs = get_all_recommendations(analysis; kwargs...)
    return isempty(recs) ? nothing : first(recs)
end



"""
    create_solver_configuration(rec::AlgorithmRecommendation, analysis::SystemAnalysis, backend_selection; 
                                 reltol::Float64=1e-6, abstol::Float64=1e-6, options...)

Create a solver configuration by instantiating the recommended algorithm with the 
optimally selected backends.
"""
function create_solver_configuration(rec::AlgorithmRecommendation, analysis::SystemAnalysis, backend_selection;
                                     reltol::Float64=1e-6,
                                     abstol::Float64=1e-6,
                                     options...)
    
    alg_constructor = rec.algorithm
    ad_backend = backend_selection.ad_backend
    lin_solver = backend_selection.linear_solver
    
    # 1. Refine AD backend based on algorithm compatibility
    # 1. Refine AD backend based on algorithm compatibility
    # Rosenbrock methods (like Rodas) currently have issues with explicit ADType objects 
    # in some OrdinaryDiffEq versions, especially when sparsity is involved.
    # Using the Boolean 'true' triggers the robust internal ForwardDiff path.

    alg_name = string(alg_constructor)
    use_bool_ad = false
    if occursin("Rodas", alg_name) || occursin("Rosenbrock", alg_name)
        use_bool_ad = true
    end

    # 2. Instantiate the algorithm with dynamic backend injection
    instance = try
        if rec.is_sundials
            # Sundials uses linear_solver keyword
            alg_constructor(linear_solver = lin_solver isa Symbol ? lin_solver : :GMRES)
        else
            # OrdinaryDiffEq uses linsolve and autodiff keywords
            # Map AD backend to Bool if required
            final_ad = use_bool_ad ? true : ad_backend
            
            if lin_solver !== nothing
                try
                    alg_constructor(linsolve = lin_solver, autodiff = final_ad)
                catch
                    try
                        alg_constructor(linsolve = lin_solver)
                    catch
                        alg_constructor()
                    end
                end
            else
                try
                    alg_constructor(autodiff = final_ad)
                catch
                    alg_constructor()
                end
            end
        end
    catch e
        @warn "Dynamic instantiation failed for $(alg_constructor), falling back to default instance. Error: $e"
        alg_constructor() 
    end
    
    return (algorithm = instance,



            reltol = reltol,
            abstol = abstol,
            options = options)
end


export get_all_recommendations, select_algorithm, create_solver_configuration

end # module AlgorithmSelector


