# Phase 2: The Performance Parity Roadmap

## Vision and Primary Objective
The next major milestone for MalthusJAX is to achieve **computational performance parity (wall-clock speed)** with EvoSAX on small-to-medium continuous optimization problems, while strictly preserving MalthusJAX's defining characteristics: **architectural modularity** and **statistical parity**.

Our goal is to prove that a fully modular, component-based evolutionary framework in JAX can execute as fast as a monolithic, flat-array implementation.

## The Challenge
MalthusJAX currently exhibits a 7–10× performance gap compared to EvoSAX on small problems (e.g., `dims=9, pop=195, gens=387`). 

Extensive profiling has isolated the root cause: **XLA HLO fusion barriers**. 
MalthusJAX's modular design passes rich PyTree dataclasses (`BasePopulation`, `RealGenome`) between pipeline phases (selection → crossover → mutation → evaluate) inside a `jax.lax.scan` loop. Calling `.replace()` on these dataclasses forces XLA to repeatedly emit tuple creation and destruction operations, preventing it from fusing the core evolutionary arithmetic into a single GPU kernel. 

EvoSAX, by contrast, operates entirely on flat `jax.Array` matrices, allowing XLA to collapse the entire generation step into a highly fused, optimal kernel (approx 1,800 HLO IR lines vs MalthusJAX's 2,247).

## Core Constraints
To declare success, the solution must satisfy three non-negotiable constraints:
1. **Preserve Modularity**: Researchers must still be able to create custom operators (Selection, Crossover, Mutation) by subclassing base classes. The Composer/TOML API must remain the primary entry point.
2. **Preserve Statistical Parity**: The new fast path must remain mathematically equivalent to the EvoSAX baseline. It must pass the existing Wilcoxon signed-rank and TOST (Two One-Sided Tests) equivalence checks.
3. **Pure JAX**: We will not write custom C++ XLA kernels. The solution must emerge from elegant JAX idioms and careful state management.

## Strategic Direction: The `FastPath` Protocol
To overcome the PyTree bottleneck without sacrificing modularity, we will implement the **`FastPath` Protocol**.

### 1. Dual-Protocol Operators
Operators will support two execution modes:
*   **Standard Mode (`__call__`)**: Takes and returns PyTrees (e.g., `BasePopulation`). Used for complex, hierarchical genomes or debugging.
*   **FastPath Mode (`apply_flat`)**: A new interface that operates directly on flat `jax.Array` matrices (e.g., `population_values`). Used for high-performance continuous optimization.

### 2. The `NativeFastEngine`
We will develop a new engine (`NativeFastEngine`) optimized for the `FastPath`. 
*   **Flat Carry**: Inside the `lax.scan` loop, the engine will only carry a flat tuple: `(population_values, fitness, rng_key, ...)`. This perfectly mirrors the EvoSAX state structure.
*   **Late Materialization**: The rich `BasePopulation` PyTree will only be constructed *once*, after the `lax.scan` loop finishes, to satisfy the `AbstractEngine` output contract.

### 3. Fused Operator Kernels
To match EvoSAX's ultimate fusion, we will explore a `FusedCrossoverMutation` operator that combines recombination and mutation into a single mathematical expression inside a single `jax.vmap`. This eliminates the intermediate arrays currently allocated between the crossover and mutation phases.

## Workflow and Validation
We will leverage the newly built **Performance Harness** (`configs/perf/h1_speed_vs_evosax.toml`) to drive this iteration loop:

1.  **Develop**: Implement `NativeFastEngine` and the `FastPath` operator interfaces.
2.  **Inspect (`make perf-hlo`)**: Ensure the generated XLA HLO graph line count drops from ~2,250 towards the 1,800 baseline, with zero `while` loops in the core generation step.
3.  **Time (`make perf-bench`)**: Validate that the wall-clock execution time drops from ~340ms to ~30ms (on Apple Silicon).
4.  **Profile (`make perf-perfetto`)**: Confirm via TensorBoard that XLA is successfully generating fused kernels (e.g., `fusion_...`) for the generation step.
5.  **Validate**: Ensure the equivalence tests still pass, proving we haven't traded correctness for speed.

## The Microbenchmark Laboratory Workflow
Before writing code in the main framework, all architectural hypotheses are tested in a scratch environment. This prevents regressions and guarantees our ideas are sound at the compiler level.

1. **Hypothesis Stage**: Write a pure JAX function in a scratch script simulating a modular interaction (e.g., testing `jax.flatten_util.ravel_pytree` inside a simulated loop).
2. **Benchmark Stage**: Compile the kernel using JAX and measure both the HLO line count and the `ms/iteration` execution speed. (e.g., our flat Native EvoSAX baseline compiles to `~2,143` HLO lines at `~0.17ms/gen`).
3. **Validation Stage**: If the candidate pattern maintains similar HLO lines without exploding into `while` loops, the hypothesis is proven.
4. **Integration Stage**: Port the proven JAX idiom into the `NativeFastEngine` and `FastPath` protocol.

## Conclusion
By cleanly separating the high-level PyTree representation (used by the Composer and the user) from the low-level flat-array execution (used inside the XLA compiled loop), we will deliver a "best of both worlds" framework: the speed of EvoSAX with the expressive modularity of MalthusJAX.
