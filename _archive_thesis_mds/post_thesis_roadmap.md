# Post-Thesis Roadmap: MalthusJAX Architecture

This document tracks technical debt, architectural refactoring, and feature expansion ideas to implement *after* the thesis is submitted, preparing MalthusJAX for a robust open-source release.

## 1. Decoupling the `benchmarking` Submodule
Currently, `results.py` is a 1,200+ line monolith that handles data storage, statistical analysis, and matplotlib plotting. This creates heavy dependencies (`scipy`, `matplotlib`) for users who just want to run headless evaluations.

**Proposed Structure:**
- `malthusjax.benchmarking.core`: Contains the pure dataclasses (`RunResult`, `ExperimentResult`, `ComparisonResult`). Zero external dependencies beyond standard Python.
- `malthusjax.benchmarking.stats`: Houses the `scipy.stats` integrations (Confidence Intervals, paired $t$-tests, speedup multipliers).
- `malthusjax.benchmarking.plotting`: Houses all `matplotlib` logic (convergence curves, boxplots, scaling matrices).

*Implementation detail:* To maintain the current ergonomic API (`comparison.plot_convergence()`), the `ComparisonResult` object can lazily import the plotting and stats modules under the hood, throwing a clean `ImportError` only if the user hasn't installed the `[viz]` or `[stats]` extras via pip.

## 2. Advanced Statistical Features
With a dedicated `stats` submodule, we can implement more rigorous evolutionary computation metrics:
- **Wilcoxon Rank-Sum Tests**: For non-parametric statistical significance testing (useful when fitness distributions are highly skewed and not normally distributed).
- **Effect Size (Cohen's $d$)**: Beyond just measuring if a speedup or fitness delta is statistically significant, calculate *how large* the effect is.
- **Bootstrapped Confidence Intervals**: For metrics where the underlying distribution is completely unknown.

## 3. Landscape & Diversity Metrics
Right now, we track `best_fitness` and `mean_fitness`. A standalone stats package would allow us to track and plot complex population dynamics:
- **Population Diversity (Variance)**: Track how quickly the population converges in the hyper-dimensional space.
- **Exploration vs Exploitation Ratio**: Measure how much of the generation is spent jumping to new local minima vs climbing down current ones.

## 4. Standalone Tooling (MalthusDash)
- Spin off the LaTeX and plotting generation into a standalone CLI tool or a lightweight local web dashboard (e.g., using Streamlit) that auto-parses the `results/` directories and provides an interactive UI to explore the JSON traces, similar to TensorBoard or Weights & Biases.

## 5. Universal Adapter Decorators (`@malthus_adapter`)
Currently, wrapping an external framework (like EvoSAX) requires manually writing a boilerplate class (`EvosaxEngineAdapter`) that manually handles JIT compilation, state-tracking, and telemetry formatting. 

We will introduce a `@malthus_adapter` decorator that standardizes this. A user will only need to define the fundamental `init_state()`, `ask()`, and `tell()` functions of an external library (like EvoJAX, QDax, or pgx). 

The decorator will automatically:
1. Wrap the logic in a `jax.lax.scan` loop.
2. Record `warmup` vs `execution` compilation timings.
3. Automatically intercept and format the output into a standard `RunResult` object. 
4. Seamlessly pipe the results directly into our new `ComparisonResult` statistical framework so that Native MalthusJAX, EvoSAX, and any future frameworks can be compared using the exact same robust $t$-tests and Speedup matrices!

## 6. Diverse Engine Ecosystem
MalthusJAX currently features the monolithic `MalthusEngine`, which is optimized for standard Generation-based Genetic Algorithms (Selection $\rightarrow$ Crossover $\rightarrow$ Mutation $\rightarrow$ Evaluate). 

We need to formalize an abstract `BaseEngine` API that allows completely different paradigms to exist as peers, while remaining 100% compatible with the `Composer` (TOML parsing) and `results.py` (telemetry).

**Future Native Engines:**
- **`SteadyStateEngine`**: Instead of replacing the entire population (generational), individuals are evaluated and replaced continuously (asynchronous).
- **`NeuroevolutionEngine` (e.g., NEAT)**: Evolves topologies, not just parameter arrays. Requires dynamic PyTree surgery.
- **`QualityDiversityEngine` (e.g., MAP-Elites)**: Instead of a singular fitness pool, it maintains an archive grid based on behavioral descriptors. The Composer TOML will need to map `fitness` to an archive rule, but the output telemetry will still seamlessly return a `RunResult` plotting archive coverage.
- **`CMAESEngine`**: For pure Covariance Matrix Adaptation, bypassing crossover entirely and focusing on massive distribution updates.

**Compatibility Layer:**
To make these radically different engines compatible with the Composer, the `engine_factory.py` will transition from a simple string switch (`if backend == "malthusjax"`) to an explicit Registry Pattern. 
Engines will register themselves with their required parameter schemas. The Composer will just route the parsed TOML blocks directly into the Engine's `__init__`, and every engine will be contractually obligated to return the standardized `RunResult` payload!

## 7. Generic Representation Layer
Currently, the `initial_population` handling inside `engine_factory.py` and `runner.py` suffers from technical debt: it hardcodes `RealPopulation.from_array(arr)`. This "distastefully" assumes all genomes are continuous real vectors, heavily restricting the framework from natively running binary, integer, or categorical permutations if an initial population is seeded.

**The Solution:**
We need to decouple the Population State from continuous matrices. 
1. **Population Factories:** The initialization layer should use a `PopulationFactory` that inspects the configuration (e.g., `genome_type = "binary"`) and dynamically instantiates `BinaryPopulation`, `PermutationPopulation`, or `RealPopulation`.
2. **PyTree Genomes:** Instead of enforcing genomes to be flat `jnp.ndarray` tensors, MalthusJAX should natively support arbitrary nested JAX PyTrees. This would allow a single genome to contain mixed data types (e.g., `{'weights': float32[100], 'activation_flags': bool[10]}`). Operators like Crossover and Mutation will map over these trees using `jax.tree_util.tree_map`, applying real-mutations to float leaves and bit-flips to boolean leaves!

## 8. Decoupling Adapters from Hardcoded Fitness Evaluators
The current `EvosaxEngineAdapter` acts as a closed-loop system: it not only generates candidates but also internally relies on EvoSAX's built-in fitness modules to evaluate them. This was an intentional "guardrail" designed to ensure pristine parity for the thesis (ensuring MalthusJAX code didn't silently interfere with EvoSAX evaluations). However, it severely limits the framework to pre-defined BBOB/Sphere benchmark functions.

**The Solution:**
External adapters should be strictly limited to the `ask()` and `tell()` candidate generation paradigm. The evaluation step must be explicitly decoupled and handed back to the MalthusJAX `BaseEvaluator` layer.
- `genomes = adapter.ask()` $\rightarrow$ (Adapter)
- `fitness = malthus_evaluator(genomes)` $\rightarrow$ (MalthusJAX)
- `adapter.tell(fitness)` $\rightarrow$ (Adapter)

This will allow users to plug an EvoSAX or QDax algorithm into a completely custom, wildly complex MalthusJAX fitness landscape (e.g., a physics simulation or a neural network environment) rather than being trapped in the external library's native benchmark suite!

## 9. [OPEN PROBLEM] The Representation Semantics Dilemma
There is an inherent "code smell" in how `BasePopulation` and `BaseGenome` currently interact. They are tightly coupled via the Struct-of-Arrays (SoA) pattern, creating ambiguity in typing (where a `BaseGenome` technically represents a batched population of genomes).

While the natural JAX solution is to abandon the `BaseGenome` class hierarchy entirely and just use generic PyTrees (e.g., `PopulationState(genes: Any, fitness: Array)`), doing so strips away **critical semantic meaning**. 
If a genome is reduced to a generic `float32[100]` PyTree leaf, the operators lose context:
- Are these 100 floats the weights of an Artificial Neural Network? (Requires specific topological crossover).
- Are they the coefficients of a Taylor Series? (Requires scaled mutation variance for higher-order terms).
- Are they just spatial coordinates?

**The Dilemma:**
If we use generic PyTrees, the operators are "dumb" and the user is forced to manually ensure mathematical correctness. If we use strict Object classes (`RealGenome`, `TaylorGenome`), we introduce heavy boilerplate and slicing complexities.

**Potential Future Explorations:**
- **Semantic Schemas (Metadata):** Passing a lightweight `schema` metadata object alongside the PyTree to instruct operators on how to mutate specific leaves.
- **Protocol/Trait typing:** Using static type-checkers to enforce operator behaviors rather than heavy inheritance.
*This remains a completely open architectural problem to be solved in MalthusJAX v2.0.*
