### DESIGN.md (File listed)

### REPO_MAP.md (File listed)

### src\Frankenstein.jl
  module Frankenstein

### src\MonsterSolver.jl
  module MonsterSolver
  function init
  function step!
  function reinit!

### src\.ipynb_checkpoints\Frankenstein-checkpoint.jl
  module Frankenstein

### src\.ipynb_checkpoints\MonsterAlgorithm-checkpoint.jl
  module MonsterAlgorithm

### src\.ipynb_checkpoints\MonsterSolver-checkpoint.jl (File listed)

### src\adaptation\adaptation.jl
  module Adaptation
  function adapt!
  function adapt!
  function adapt!
  function adapt!
  function adapt!
  function adapt!
  function adapt!

### src\adaptation\convergence_adaptation.jl
  module ConvergenceAdaptation
  function adapt!

### src\adaptation\hybrid_adaptation.jl
  module HybridAdaptation
  function adapt!

### src\adaptation\memory_adaptation.jl
  module MemoryAdaptation
  function adapt!

### src\adaptation\parallel_adaptation.jl
  module ParallelAdaptation
  function adapt!

### src\adaptation\performance_adaptation.jl
  module Adaptation
  function adapt!

### src\adaptation\stability_adaptation.jl
  module StabilityAdaptation
  function adapt!

### src\adaptation\.ipynb_checkpoints\adaptation-checkpoint.jl (File listed)

### src\adaptation\.ipynb_checkpoints\convergence_adaptation-checkpoint.jl
  module ConvergenceAdaptation
  function adapt!

### src\adaptation\.ipynb_checkpoints\hybrid_adaptation-checkpoint.jl (File listed)

### src\adaptation\.ipynb_checkpoints\memory_adaptation-checkpoint.jl
  module MemoryAdaptation
  function adapt!

### src\adaptation\.ipynb_checkpoints\parallel_adaptation-checkpoint.jl
  module ParallelAdaptation
  function adapt!

### src\adaptation\.ipynb_checkpoints\performance_adaptation-checkpoint.jl
  module Adaptation
  function adapt!

### src\adaptation\.ipynb_checkpoints\stability_adaptation-checkpoint.jl
  module StabilityAdaptation
  function adapt!

### src\analysis\analysis.jl
  module Analysis
  function analyze_system_structure
  function needs_analysis_update!

### src\analysis\condition_analysis.jl
  function compute_condition_number
  function update_condition_number!

### src\analysis\coupling_analysis.jl
  function compute_coupling_strength
  function update_coupling_strength!

### src\analysis\sparsity_analysis.jl
  module Analysis.Sparsity
  function detect_sparsity_patterns

### src\analysis\stiffness_analysis.jl
  module Analysis.Stiffness
  function gershgorin_spectral_bound
  function initial_stiffness_estimate
  function update_stiffness!

### src\analysis\timescale_analysis.jl
  function compute_timescales
  function update_timescales!

### src\analysis\untitled.jl
  function compute_coupling_strength

### src\analysis\.ipynb_checkpoints\analysis-checkpoint.jl
  module Analysis
  function analyze_system_structure
  function needs_analysis_update!

### src\analysis\.ipynb_checkpoints\condition_analysis-checkpoint.jl (File listed)

### src\analysis\.ipynb_checkpoints\coupling_analysis-checkpoint.jl
  function compute_coupling_strength

### src\analysis\.ipynb_checkpoints\sparsity_analysis-checkpoint.jl
  function detect_sparsity_patterns

### src\analysis\.ipynb_checkpoints\stiffness_analysis-checkpoint.jl
  module Analysis.Stiffness
  function gershgorin_spectral_bound
  function initial_stiffness_estimate
  function update_stiffness!

### src\analysis\.ipynb_checkpoints\timescale_analysis-checkpoint.jl
  function compute_timescales
  function update_timescales!

### src\backends\AD_interface.jl
  function evaluate_backend_performance
  function is_backend_suitable

### src\backends\backends.jl
  module Backends
  function BackendPerformanceMetrics{T}

### src\backends\backend_selector.jl
  function choose_backend
  function evaluate_backend_score
  function generate_rationale

### src\backends\enzyme_backend.jl
  function configure_enzyme
  function is_enzyme_suitable

### src\backends\finite_difference.jl
  function configure_finite_diff
  function adaptive_stepsize

### src\backends\hybrid_backend.jl
  function configure_hybrid
  function switch_backend
  function ADTypes.jacobian

### src\backends\linsolve_interface.jl
  function analyze_matrix
  function check_diagonal_dominance
  function estimate_bandwidth
  function select_linear_solver
  function select_solver_by_properties
  function select_small_problem_solver
  function select_medium_problem_solver
  function select_large_problem_solver
  function select_sparse_solver
  function select_dense_solver

### src\backends\sparse_forwarddiff.jl
  function configure_sparse_forwarddiff
  function detect_sparsity_pattern
  function optimize_chunk_size

### src\backends\symbolic_backend.jl
  function configure_symbolic
  function ADTypes.jacobian
  function is_symbolic_suitable

### src\backends\.ipynb_checkpoints\AD_interface-checkpoint.jl
  function evaluate_backend_performance
  function is_backend_suitable

### src\backends\.ipynb_checkpoints\backends-checkpoint.jl
  module Backends
  function BackendPerformanceMetrics{T}

### src\backends\.ipynb_checkpoints\backend_selector-checkpoint.jl
  function choose_backend
  function evaluate_backend_score
  function generate_rationale

### src\backends\.ipynb_checkpoints\enzyme_backend-checkpoint.jl
  function configure_enzyme
  function is_enzyme_suitable

### src\backends\.ipynb_checkpoints\finite_difference-checkpoint.jl
  function configure_finite_diff
  function adaptive_stepsize

### src\backends\.ipynb_checkpoints\hybrid_backend-checkpoint.jl
  function configure_hybrid
  function switch_backend
  function ADTypes.jacobian

### src\backends\.ipynb_checkpoints\linsolve_interface-checkpoint.jl
  function analyze_matrix
  function check_diagonal_dominance
  function estimate_bandwidth
  function select_linear_solver
  function select_solver_by_properties
  function select_small_problem_solver
  function select_medium_problem_solver
  function select_large_problem_solver
  function select_sparse_solver
  function select_dense_solver

### src\backends\.ipynb_checkpoints\sparse_forwarddiff-checkpoint.jl
  function configure_sparse_forwarddiff
  function detect_sparsity_pattern
  function optimize_chunk_size

### src\backends\.ipynb_checkpoints\symbolic_backend-checkpoint.jl
  function configure_symbolic
  function ADTypes.jacobian
  function is_symbolic_suitable

### src\core\core.jl
  module Core
  function Frankenstein
  function solve
  function record_step
  function SystemAnalysis{T}
  function PerformanceProfile{T}

### src\core\.ipynb_checkpoints\core-checkpoint.jl
  module Core
  function Frankenstein
  function SystemAnalysis{T}
  function PerformanceProfile{T}

### src\monitoring\monitoring.jl
  module Monitoring

### src\preconditioning\preconditioning.jl
  module Preconditioning

### src\solvers\adaptive_solvers.jl
  module AdaptiveSolvers
  function build_adaptive_solver_catalogue
  function get_adaptive_recommendations

### src\solvers\algorithm_selector.jl
  module AlgorithmSelector
  function get_all_recommendations
  function select_algorithm
  function create_solver_configuration

### src\solvers\base_types.jl
  function AlgorithmRecommendation
  function classify_stiffness
  function classify_system_size
  function classify_accuracy_level
  function has_multiscale_behavior
  function is_well_conditioned
  function requires_sparse_handling
  function is_applicable
  function compute_adjusted_priority

### src\solvers\composite_solvers.jl
  function is_mixed_stiffness_problem
  function estimate_explicit_fraction
  function get_composite_recommendations
  function analyze_splitting_potential
  function configure_composite_solver
  function recommend_composite_solver
  function suggest_problem_splitting

### src\solvers\explicit_solvers.jl
  function create_explicit_algorithms
  function get_explicit_recommendations
  function select_best_explicit

### src\solvers\multiscale_solvers.jl
  function analyze_timescale_separation
  function cluster_timescales
  function classify_multiscale_problem
  function get_multiscale_recommendations
  function estimate_multiscale_efficiency
  function configure_multiscale_solver
  function recommend_multiscale_solver
  function analyze_multiscale_structure

### src\solvers\solvers.jl
  module Solvers
  function select_best_algorithm

### src\solvers\sparse_solvers.jl
  module SparseSolvers
  function build_sparse_solver_catalogue
  function get_sparse_recommendations

### src\solvers\specialty_solvers.jl
  module SpecialtySolvers
  function build_specialty_solver_catalogue
  function get_specialty_recommendations

### src\solvers\stiff_solvers.jl
  function get_stiff_recommendations
  function get_linear_solver_recommendation
  function configure_stiff_solver
  function recommend_stiff_solver

### src\solvers\.ipynb_checkpoints\adaptive_solvers-checkpoint.jl
  module AdaptiveSolvers
  function build_adaptive_solver_catalogue
  function get_adaptive_recommendations

### src\solvers\.ipynb_checkpoints\algorithm_selector-checkpoint.jl
  module AlgorithmSelector
  function get_all_recommendations
  function select_algorithm
  function create_solver_configuration

### src\solvers\.ipynb_checkpoints\base_types-checkpoint.jl
  function AlgorithmRecommendation
  function classify_stiffness
  function classify_system_size
  function classify_accuracy_level
  function has_multiscale_behavior
  function is_well_conditioned
  function requires_sparse_handling
  function is_applicable
  function compute_adjusted_priority

### src\solvers\.ipynb_checkpoints\composite_solvers-checkpoint.jl
  function is_mixed_stiffness_problem
  function estimate_explicit_fraction
  function get_composite_recommendations
  function analyze_splitting_potential
  function configure_composite_solver
  function recommend_composite_solver
  function suggest_problem_splitting

### src\solvers\.ipynb_checkpoints\explicit_solvers-checkpoint.jl
  function create_explicit_algorithms
  function get_explicit_recommendations
  function select_best_explicit

### src\solvers\.ipynb_checkpoints\multiscale_solvers-checkpoint.jl
  function analyze_timescale_separation
  function cluster_timescales
  function classify_multiscale_problem
  function get_multiscale_recommendations
  function estimate_multiscale_efficiency
  function configure_multiscale_solver
  function recommend_multiscale_solver
  function analyze_multiscale_structure

### src\solvers\.ipynb_checkpoints\solvers-checkpoint.jl
  module Solvers
  function select_best_algorithm

### src\solvers\.ipynb_checkpoints\sparse_solvers-checkpoint.jl
  module SparseSolvers
  function build_sparse_solver_catalogue
  function get_sparse_recommendations

### src\solvers\.ipynb_checkpoints\specialty_solvers-checkpoint.jl
  module SpecialtySolvers
  function build_specialty_solver_catalogue
  function get_specialty_recommendations

### src\solvers\.ipynb_checkpoints\stiff_solvers-checkpoint.jl
  function get_stiff_recommendations
  function get_linear_solver_recommendation
  function configure_stiff_solver
  function recommend_stiff_solver

### src\splitting\splitting.jl
  module Splitting

### src\utilities\jacobians.jl
  function finite_difference_jac
  function compute_jacobian

### src\utilities\logging.jl
  function log_monster_info

### src\utilities\utilities.jl
  module Utilities

### src\utilities\.ipynb_checkpoints\jacobians-checkpoint.jl
  function finite_difference_jac
  function compute_jacobian

### src\utilities\.ipynb_checkpoints\logging-checkpoint.jl
  function log_monster_info

### src\utilities\.ipynb_checkpoints\utilities-checkpoint.jl
  module Utilities