# Frankenstein.jl 🧠⚡

**The Monster Solver — Compositional, Adaptive, and Nearly Sentient.**

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
Frankenstein.jl is a "meta-solver" for the Julia SciML ecosystem. Built from the best stiff and non-stiff integrators (Tsit5, Rodas5, KenCarp4, FBDF, etc.), it dynamically assembles the ideal solver strategy for your problem using intelligent heuristics and modular adaptation.

Stop wasting time picking algorithms. Just call the Monster.

---

## 🚀 "It's Alive!" — The Pitch

Are you tired of manually testing twenty different solvers for your ODE system? Is your problem stiff in some regions but non-stiff in others? 

**Frankenstein.jl** handles the complexity so you don't have to. It is a "black-box" solver that:
- **Automatically detects stiffness** and switches integrators mid-simulation.
- **Analyzes problem structure** (sparsity, scaling, coupling) to pick the optimal backend.
- **Exploits cutting-edge AD** (Enzyme, ForwardDiff) and symbolic tools (ModelingToolkit).
- **Just Works™** with the standard `DifferentialEquations.jl` interface.

```julia
using Frankenstein, DifferentialEquations

prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Monster()) # The Monster takes care of the rest
```

---

## 🛠 Features & Capabilities

### 1. Dynamic Method Switching
Frankenstein is built on the principle of "Algorithm Stitching." It monitors the stiffness ratio and convergence rates in real-time. When it detects a transition from non-stiff to stiff dynamics, it pauses, saves the state, swaps the integrator, and resumes—all within a single `solve` call.

### 2. Deep Structural Analysis
Before and during integration, the "Brain" analyzes your system:
- **Stiffness Analysis:** Estimates spectral radius and eigenvalue separation.
- **Sparsity Analysis:** Inspects Jacobian density and bandwidth to recommend CSC/CSR structures or JFNK (Jacobian-Free Newton-Krylov) methods.
- **Timescale Separation:** Detects multiscale behavior to suggest HMM or specialized splitting techniques.

### 3. High-Performance Backends
Frankenstein interfaces with the best in the business:
- **AD Support:** Integrated with Enzyme for state-of-the-art reverse-mode and ForwardDiff for robust forward-mode.
- **Symbolics:** Leverages Symbolics.jl for analytical Jacobians and simplification.
- **Linear Solvers:** Unified interface for dense, sparse, and iterative (GMRES, BiCGSTAB) solvers.

### 4. Robust Event Handling
Supports continuous and discrete callbacks via an internal `EventManager`. It handles root-finding for zero-crossings and applies state jumps while maintaining numerical stability.

---

## 📂 Project Architecture

| Component | Purpose |
| :--- | :--- |
| **`Core`** | Central utilities and shared abstractions ([core.jl](file:///c:/Users/jelte/projects/Frankenstein/src/core/core.jl)). |
| **`Analysis`** | Tools for identifying stiffness, sparsity, and timescale separation. |
| **`Adaptation`** | The "Brain" that manages memory-based and performance-aware strategy switching. |
| **`Solvers`** | A massive library of explicit, implicit, and multiscale integrators. |
| **`Backends`** | Unified interface for Automatic Differentiation and Linear Algebra. |

---

## 📖 Usage Guide

### Basic Usage
The primary entry point is the `Monster()` algorithm:
```julia
sol = solve(prob, Monster())
```

### Advanced Configuration
You can provide hints to the Monster to guide its heuristics:
```julia
sol = solve(prob, Monster(), 
            prefer_stability = true, 
            ad_available = [AutoForwardDiff(), AutoEnzyme()], 
            sparse = true)
```

### Callbacks and Events
Full compatibility with SciML callbacks:
```julia
cb = ContinuousCallback(condition, affect!)
sol = solve(prob, Monster(), callback=cb)
```

---

## 🏗 Roadmap & Status

While the "Brain" of the Monster is highly capable of analyzing problem structures, the current focus is on unlocking massive scale via preconditioning and specialized splitting.

| Phase | Milestone | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1** | Core Solver Interface & Routing | ✅ Complete | The `Monster()` wrapper is fully integrated with `SciMLBase` and standard ODE definitions. |
| **2** | Structural System Analysis | ✅ Complete | A-priori detection of stiffness, sparsity density, and coupling architectures. |
| **3** | Dynamic Backend Hot-Swapping | ✅ Complete | The "Surgery" system is now robust against AD dimension mismatches. **100% Native Julia implementation** (Sundials-free). |
| **4** | Hybrid Dynamic Adaptation | ✅ Complete | Real-time transitions powered by `light_pulse` heartbeat diagnostics and multi-strategy `AdaptationController`. |
| **5** | Preconditioning Pipeline | 🚧 In Progress | Implementing ILU and AMG preconditioners to unlock large-scale GMRES/Krylov performance. |
| **6** | Operator Splitting & Multiscale | 🔴 Planned | Future support for separating stiff chemistry from non-stiff advection in multiscale PDE problems. |

---

## Latest Benchmarks

Obvious that it will be slower, the solver can find solutions which reduce number of steps required for problems of greater complexity. This could have use in hours long simulations with intermittent regions of varying stiffness and sparsity, and situations with instabilities where robustness is preferred over speed.

```julia
PS C:\Users\jelte\projects\Frankenstein> julia --project=. test/test_pde_suite.jl                                                          
[ Info: --- Benchmarking: 2D Heat (100% Diffusion) ---                                                                                     
[ Info: [Frankenstein Analysis] System Size: 2500 | Sparse: true | Density: 0.2%                                                           
[ Info: [Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling.                                         
[ Info: [Frankenstein] Initializing with TRBDF2{0, AutoSparse{AutoFiniteDiff{Val{:forward}, Val{:forward}, Val{:hcentral}, Nothing, Nothing, Bool}, Frankenstein.Backends.PrecomputedSparsityDetector{SparseMatrixCSC{Float64, Int64}}, SparseMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings.NaturalOrder}}}, KLUFactorization, NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}, Nothing}, typeof(OrdinaryDiffEqCore.DEFAULT_PRECS), Val{:forward}(), true, nothing, typeof(OrdinaryDiffEqCore.trivial_limiter!)}
[ Info: [Frankenstein] Backend selection: Sparse AD: Exploiting 0.2% density for PDE-optimal scaling.
┌ Warning: Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information
└ @ SciMLBase C:\Users\jelte\.julia\packages\SciMLBase\wfZCo\src\retcodes.jl:448
  [Native] Time: 0.8672s | Steps: 98 | Retcode: Success
  [Monster] Time: 10.279s | Steps: 21 | Retcode: Success
  >> Native is 11.85x FASTER
------------------------------------------
Test Summary:    | Pass  Total   Time
2D Heat Equation |    3      3  32.9s
[ Info: --- Benchmarking: 2D Wave Equation ---
[ Info: [Frankenstein Analysis] System Size: 1250 | Sparse: true | Density: 0.23%
[ Info: [Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling.
[ Info: [Frankenstein] Initializing with TRBDF2{0, AutoSparse{AutoFiniteDiff{Val{:forward}, Val{:forward}, Val{:hcentral}, Nothing, Nothing, Bool}, Frankenstein.Backends.PrecomputedSparsityDetector{SparseMatrixCSC{Float64, Int64}}, SparseMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings.NaturalOrder}}}, KLUFactorization, NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}, Nothing}, typeof(OrdinaryDiffEqCore.DEFAULT_PRECS), Val{:forward}(), true, nothing, typeof(OrdinaryDiffEqCore.trivial_limiter!)}
[ Info: [Frankenstein] Backend selection: Sparse AD: Exploiting 0.23% density for PDE-optimal scaling.
┌ Error: [Frankenstein] Step failed with error: OrdinaryDiffEqDifferentiation.FirstAutodiffJacError(DimensionMismatch("all inputs to eachindex must have the same indices, got Base.OneTo(4900) and Base.OneTo(3650)"))
└ @ Frankenstein.MonsterSolver C:\Users\jelte\projects\Frankenstein\src\MonsterSolver.jl:137
┌ Warning: [Frankenstein] Recovery attempt 1/5...
└ @ Frankenstein.MonsterSolver C:\Users\jelte\projects\Frankenstein\src\MonsterSolver.jl:145
┌ Warning: [Frankenstein] Sparse AD mismatch detected. Disabling backend and performing Surgery...
└ @ Frankenstein.MonsterSolver C:\Users\jelte\projects\Frankenstein\src\MonsterSolver.jl:150
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=0.0
[ Info: [Frankenstein] Results: Stiffness=1.4 | Coupling=1.0
[ Info: [Frankenstein] Attempting recovery with: Enzyme: High-performance reverse-mode scaling for large dense systems (n=1250).
┌ Warning: Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information
└ @ SciMLBase C:\Users\jelte\.julia\packages\SciMLBase\wfZCo\src\retcodes.jl:448
  [Native] Time: 0.3511s | Steps: 127 | Retcode: Success
  [Monster] Time: 32.078s | Steps: 51 | Retcode: Success
  >> Native is 91.36x FASTER
------------------------------------------
Test Summary:    | Pass  Total   Time
2D Wave Equation |    1      1  41.4s
[ Info: --- Benchmarking: Burgers' Equation ---
[ Info: [Frankenstein Analysis] System Size: 900 | Sparse: true | Density: 0.54%
[ Info: [Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling.
[ Info: [Frankenstein] Initializing with FBDF{5, 0, AutoSparse{AutoFiniteDiff{Val{:forward}, Val{:forward}, Val{:hcentral}, Nothing, Nothing, Bool}, Frankenstein.Backends.PrecomputedSparsityDetector{SparseMatrixCSC{Float64, Int64}}, SparseMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings.NaturalOrder}}}, KLUFactorization, NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}, Nothing}, typeof(OrdinaryDiffEqCore.DEFAULT_PRECS), Val{:forward}(), true, nothing, Nothing, Nothing, typeof(OrdinaryDiffEqCore.trivial_limiter!)}
[ Info: [Frankenstein] Backend selection: Sparse AD: Exploiting 0.54% density for PDE-optimal scaling.
[ Info: [Frankenstein] Pulse detected anomaly at t=5.117154863647436e-9. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=5.117154863647436e-9
[ Info: [Frankenstein] Results: Stiffness=66.37 | Coupling=0.5
[ Info: [Frankenstein] Pulse detected anomaly at t=1.6404369680573245e-6. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=1.6404369680573245e-6
[ Info: [Frankenstein] Results: Stiffness=66.28 | Coupling=0.5
[ Info: [Frankenstein] Pulse detected anomaly at t=0.5. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=0.5
[ Info: [Frankenstein] Results: Stiffness=64.75 | Coupling=0.5
┌ Warning: Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information
└ @ SciMLBase C:\Users\jelte\.julia\packages\SciMLBase\wfZCo\src\retcodes.jl:448
  [Native] Time: 0.0452s | Steps: 8 | Retcode: Success
  [Monster] Time: 3.3905s | Steps: 49 | Retcode: Success
  >> Native is 75.09x FASTER
------------------------------------------
Test Summary:     | Pass  Total   Time
Burgers' Equation |    1      1  11.0s
[ Info: --- Benchmarking: ADR Equation ---
[ Info: [Frankenstein Analysis] System Size: 625 | Sparse: true | Density: 0.77%
[ Info: [Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling.
[ Info: [Frankenstein] Initializing with FBDF{5, 0, AutoSparse{AutoFiniteDiff{Val{:forward}, Val{:forward}, Val{:hcentral}, Nothing, Nothing, Bool}, Frankenstein.Backends.PrecomputedSparsityDetector{SparseMatrixCSC{Float64, Int64}}, SparseMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings.NaturalOrder}}}, KLUFactorization, NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}, Nothing}, typeof(OrdinaryDiffEqCore.DEFAULT_PRECS), Val{:forward}(), true, nothing, Nothing, Nothing, typeof(OrdinaryDiffEqCore.trivial_limiter!)}
[ Info: [Frankenstein] Backend selection: Sparse AD: Exploiting 0.77% density for PDE-optimal scaling.
[ Info: [Frankenstein] Pulse detected anomaly at t=8.575767622843089e-10. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=8.575767622843089e-10
[ Info: [Frankenstein] Results: Stiffness=49.04 | Coupling=0.62
[ Info: [Frankenstein] Pulse detected anomaly at t=1.37825615553784e-6. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=1.37825615553784e-6
[ Info: [Frankenstein] Results: Stiffness=48.97 | Coupling=0.62
┌ Warning: Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information
└ @ SciMLBase C:\Users\jelte\.julia\packages\SciMLBase\wfZCo\src\retcodes.jl:448
  [Native] Time: 0.0437s | Steps: 14 | Retcode: Success
  [Monster] Time: 2.3966s | Steps: 40 | Retcode: Success
  >> Native is 54.78x FASTER
------------------------------------------
Test Summary:                | Pass  Total   Time
Advection‑Diffusion‑Reaction |    1      1  10.0s
[ Info: --- Benchmarking: Gray-Scott Equation ---
[ Info: [Frankenstein Analysis] System Size: 1800 | Sparse: true | Density: 0.54%
[ Info: [Frankenstein] Injecting Sparse FiniteDiff and Greedy Coloring for robust sparse handling.
[ Info: [Frankenstein] Initializing with TRBDF2{0, AutoSparse{AutoFiniteDiff{Val{:forward}, Val{:forward}, Val{:hcentral}, Nothing, Nothing, Bool}, Frankenstein.Backends.PrecomputedSparsityDetector{SparseMatrixCSC{Float64, Int64}}, SparseMatrixColorings.GreedyColoringAlgorithm{:direct, 1, Tuple{SparseMatrixColorings.NaturalOrder}}}, KLUFactorization, NLNewton{Rational{Int64}, Rational{Int64}, Rational{Int64}, Nothing}, typeof(OrdinaryDiffEqCore.DEFAULT_PRECS), Val{:forward}(), true, nothing, typeof(OrdinaryDiffEqCore.trivial_limiter!)}
[ Info: [Frankenstein] Backend selection: Sparse AD: Exploiting 0.54% density for PDE-optimal scaling.
[ Info: [Frankenstein] Pulse detected anomaly at t=0.09139634988810524. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=0.09139634988810524
[ Info: [Frankenstein] Results: Stiffness=1047.95 | Coupling=0.45
[ Info: [Frankenstein] Pulse detected anomaly at t=0.10280659926589276. Performing heavy diagnostics...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=0.10280659926589276
[ Info: [Frankenstein] Results: Stiffness=1030.19 | Coupling=0.45
[ Info: [Frankenstein] Pulse detected anomaly at t=0.1522969384134703. Performing heavy diagnostics...
...
[ Info: [Frankenstein] 🛰️ Heavy Diagnostic Triggered at t=5.0
[ Info: [Frankenstein] Results: Stiffness=1027.05 | Coupling=0.45
┌ Warning: Backwards compatibility support of the new return codes to Symbols will be deprecated with the Julia v1.9 release. Please see https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes for more information
└ @ SciMLBase C:\Users\jelte\.julia\packages\SciMLBase\wfZCo\src\retcodes.jl:448
  [Native] Time: 1.5449s | Steps: 141 | Retcode: Success
  [Monster] Time: 4.726s | Steps: 33 | Retcode: Success
  >> Native is 3.06x FASTER
------------------------------------------
Test Summary:                 | Pass  Total   Time
Gray‑Scott Reaction‑Diffusion |    1      1  13.7s
PS C:\Users\jelte\projects\Frankenstein> ```

---

## 📜 License & Credit

Built by **Jelterminator**. Inspired by the goal of becoming the "scikit-learn of ODE solvers."

Licensed under the **MIT License**. 

> "Nothing is so painful to the human mind as a great and sudden change... unless that change is handled by Frankenstein.jl." — *Victor Frankenstein (paraphrased)*
