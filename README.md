# Frankenstein.jl 🧠⚡

**The Monster Solver — Compositional, Adaptive, and Nearly Sentient.**

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

## 📜 License & Credit

Built by **Jelterminator**. Inspired by the goal of becoming the "scikit-learn of ODE solvers."

Licensed under the **MIT License**. 

> "Nothing is so painful to the human mind as a great and sudden change... unless that change is handled by Frankenstein.jl." — *Victor Frankenstein (paraphrased)*
