# Show and Tell: MalthusJAX — A Unified JAX Framework for Evolutionary Computation

Hi everyone! I’m sharing **MalthusJAX**, an open-source framework for hardware-accelerated Evolutionary Computation natively in JAX. 

This is my first major open-source project in this domain, and I'm sharing it here primarily to gather feedback, critiques, and suggestions from the community on the overall architecture, API design, and benchmarking methodology.

---

## 1. On-Device Execution & Scaling Behavior
At its core, MalthusJAX keeps the entire evolutionary loop—from mutation and crossover to evaluation and Pareto sorting—strictly on-device within `jax.jit` and `jax.vmap`. This eliminates CPU-GPU memory transfers during evolution. 

*Note on throughput:* Raw evaluation throughput alone isn't very meaningful without contextualized comparisons against other libraries (which is why we built the unified adapter and parity benchmarking suite below). The throughput numbers primarily confirm that execution is running entirely on-device without host-device synchronization bottlenecks:
- On an NVIDIA H100 GPU benchmark, throughput reaches **~64,500 evaluations per second** on small population benchmarks (50-dimensional Sphere problem), and scales into millions of evaluations per second for larger population sizes.
- **Dimensionality Scaling:** Scaling problem dimensionality 10× (from `dim=5` to `dim=50`) increases execution time by just **~0.08 seconds** (0.845s → 0.931s over 250–300 generations), demonstrating near-constant $O(1)$ GPU compute scaling across problem dimensions.

---

## 2. Ecosystem Adapters & Environment Integrations
Rather than re-implementing every existing evolutionary algorithm or evaluation environment from scratch, MalthusJAX provides unified wrappers for established JAX libraries:

**Algorithm Adapters:**
- **[EvoSAX](https://github.com/RobertTLange/evosax)** for Evolution Strategies (CMA-ES, OpenES, SimpleGA).
- **[QDAX](https://github.com/adaptive-intelligent-robotics/QDax)** for Quality-Diversity algorithms (MAP-Elites).
- **[TensorNEAT](https://github.com/EMI-Group/tensorneat)** for Neuroevolution and variable-topology neural networks.

**Environment & Fitness Evaluators:**
- **[Brax](https://github.com/google/brax)** for hardware-accelerated physics and locomotion robotics simulations.
- **[Jumanji](https://github.com/instadeepai/jumanji)** for combinatorial optimization and decision-making environments.
- **[Gymnax](https://github.com/RobertTLange/gymnax)** for standard RL control benchmarks.
- **BBOB-JAX** for vectorized black-box optimization functions.

**How adapters work under the hood:**
To avoid per-generation conversion overhead across different state representations (population vectors, repertoire archives, graph structures), MalthusJAX unifies upstream states into a single internal representation *once* during initialization. The rest of the evolutionary loop is fused into a single `jax.jit` compiled graph.

---

## 3. Vectorized Island Models
MalthusJAX includes a vectorized implementation of island models (`RingTopologyIsland`). Instead of using Python multiprocessing or networking overhead, `jax.vmap` vectorizes the entire evolutionary engine across independent island populations on-device. Migration between islands is handled asynchronously within the JAX execution graph via GPU permutation matrices.

---

## 4. Native Multi-Objective Evolution (NSGA-II)
The library includes a JIT-compiled implementation of NSGA-II. Non-dominated Pareto sorting and crowding distance calculations are fused directly into the execution graph, allowing fast multi-objective sorting on-device.

---

## 5. Focus on Parity & Fair Benchmarking
A key priority during development was ensuring fair comparisons across different algorithms:
- **Parity Verification:** We built automated test suites comparing MalthusJAX adapters side-by-side with native upstream implementations on identical random seeds.
- **Standardized Benchmark Outputs:** Our `BenchmarkRunner` wraps all adapters and native engines to return standardized `ExperimentResult` objects, capturing identical history metrics, serialization formats, and wall-clock timings across all algorithms.

**Maintenance & Version Compatibility:**
To ensure stability as upstream libraries evolve, we maintain a compatibility matrix (pinning versions such as EvoSAX 0.1.5) tested continuously via automated parity pipelines.

---

## 🧱 Layered Architecture & Unified Benchmarking

MalthusJAX is decoupled into distinct layers so users can interact at whichever level of abstraction fits their workflow:

1. **Layer 1: Core Primitives & State (`malthusjax.core`)**
   Pure functional JAX PyTree data structures (`Population`, `BaseGenome`, `Evaluator`). No magic or hidden state—just immutable JAX arrays and dataclasses:
   - **`BaseGenome` Subclasses:** Represent genetic encodings (`RealGenome`, `BinaryGenome`, `PermutationGenome`, `TensorNeatGenome`) as PyTrees wrapping typed JAX arrays for parameters, gene bounds, and structural topology matrices.
   - **`Population` Container:** A vectorized PyTree container that pairs individual genome arrays alongside population-level state. It manages genome PyTrees together with fitness matrices (`fitness`), multi-objective targets (`objectives`), PRNGKeys, generation counters, and flexible metadata dictionaries (`meta`) that pass problem-specific context smoothly across JAX `jit`/`vmap` boundaries.
   - **`Evaluator`:** Stateless functional contracts mapping PyTree populations to evaluated fitness tensors.

2. **Layer 2: Pure Functional Operators (`malthusjax.operators`)**
   Stateless JAX functions for mutation, crossover, selection, and emissions that can be used directly inside custom JAX loops:
   - **Mutation, Crossover & Selection:** Pure functional primitives (e.g., `gaussian_mutation`, `blend_crossover`, `tournament_selection`) operating directly on JAX arrays and PyTree populations.
   - **Emitters & Injection Operators:** Emitter abstractions (`BaseEmitter`, `EmitterState`) that encapsulate generation dynamics (like QD archive emission or genetic mixing) and injection operators for seeding external solutions into PyTree populations.
   - **PRNG Key Management:** Every operator takes explicit JAX `PRNGKey`s and returns mutated PyTrees alongside updated keys, maintaining 100% functional purity and reproducibility under `jax.vmap`/`jax.jit`.

3. **Layer 3: Native Engine Loops (`malthusjax.engine`)**
   Fully functional, JIT-compilable evolutionary engines written in pure Python/JAX:
   - **Single-Population & Multi-Objective Engines:** Engines (`GeneticEngine`, `NSGA2Engine`, `MapElites`) implementing the complete evolutionary lifecycle (`init`, `step`, `tell`, `ask`) entirely on-device.
   - **Vectorized Island Topologies:** Multi-population models (`RingTopologyIsland`) that use `jax.vmap` to run $N$ independent engines in parallel on GPU/TPU, performing asynchronous cross-island migrant swaps via JAX permutation matrices.
   - **Decoupled Loop Execution:** Exposes pure step functions (`engine.step(state) -> (state, metrics)`) compatible with `jax.lax.fori_loop` or `jax.lax.scan` for zero-overhead multi-generation runs.

4. **Layer 4: Composer & Adapters (`malthusjax.composer`)**
   Declarative configuration layer for zero-code experiment reproducibility. **This layer is completely optional:**
   - **Unified Adapter Protocol:** Bridges external frameworks (EvoSAX, QDAX, TensorNEAT) by translating upstream state dictionaries and strategy objects into unified MalthusJAX PyTree states during initialization.
   - **Dynamic Catalog Registry:** String-based catalog lookups (`OperatorCatalog`, `EngineCatalog`) that resolve declarative string specs (e.g., `"tournament:tournament_size=3"`, `"gaussian:mutation_rate=0.1"`) into configured operator factories.
   - **Declarative TOML Integration:** Parses version-controlled TOML experiment definitions to automatically wire together genomes, fitness evaluators, operators, and engine loops without requiring custom Python glue code.

5. **Layer 5: Unified Benchmarking & Statistics (`malthusjax.stats` & `malthusjax.benchmarking`)**
   Standardized evaluation and statistical testing layer:
   - **Standardized Execution Harness:** The `BenchmarkRunner` executes any engine or framework adapter across parameterized seeds and problem configurations, collecting identical structured `ExperimentResult` data.
   - **Automated Parity & Statistical Testing:** The `stats` sub-package computes non-parametric tests (Wilcoxon signed-rank, TOST equivalence), effect sizes (Cohen's $d_z$), OLS log-log scaling regressions, and Bland-Altman agreement plots to rigorously evaluate algorithm performance without manual data wrangling.

---

### 💻 Code Example: Vectorized Island Model

```python
# Create 4 independent Islands, migrating 10 individuals every 50 generations
island_model = RingTopologyIsland(
    base_engine=base_engine,
    num_islands=4,
    migration_interval=50,
    num_migrants=10,
)

# Compile and run the entire distributed model on GPU/TPU
global_state = jax.jit(island_model.init)(key)
final_state = jax.lax.fori_loop(0, 2000, lambda _, s: island_model.step(s)[0], global_state)
```

---

### 🧪 Reproducing Parity & Benchmark Suites (H1, H2, H3)

We provide `Makefile` commands to run side-by-side evaluation pipelines for testing parity, operator overhead, and precision scaling:

#### 🎯 H1: Algorithm Parity
Tests ground-truth parity against native libraries on identical random seeds:

1. **EvoSAX:**
   ```bash
   make h1-parity-full
   ```
   *Runs a closed-loop comparison between the MalthusJAX wrapper and native EvoSAX `SimpleGA`.*

2. **Quality-Diversity (QDAX):**
   ```bash
   make h1-parity-qdax-full
   ```
   *Runs a MAP-Elites parity check comparing MalthusJAX's native `MapElitesStrategy` directly against QDAX, monitoring QD-Score, Max Fitness, and Archive Grid Coverage.*

3. **Neuroevolution (TensorNEAT):**
   ```bash
   make h1-parity-tensorneat-full
   ```
   *Runs a parity check on XOR neural architecture search (P=1024, G=500, 10 seeds), verifying that variable-topology mutation, innovation tracking, and fitness trajectories match pure TensorNEAT.*

#### 🧩 H2: Structural Ablation
Isolates operator overhead by systematically replacing wrapped operators with native MalthusJAX equivalents:
```bash
make h2-ablation-full
```

#### 🧮 H3: Precision Scaling
Tests execution stability and diversity retention across `float32`, `bfloat16`, and `float16`:
```bash
make h3-representation-full
```

#### 🔧 Optional TOML Configuration Grammar
Experiments can also be specified via TOML configuration files:

```toml
[engine]
adapter = "evosax"
pop_size = 1024
generations = 500
seed = 42

[genome]
type = "binary"
size = 256

[operators]
mutation = "binary_mutation"
crossover = "uniform_crossover"
selection = "tournament"

[fitness]
callable = "my_project.fitness.my_obj"
objective = "maximize"
```

#### 📉 Advanced: LHS Benchmarking & Statistical Analysis (Beta)
For sweeps across function spaces, Latin Hypercube Sampling (LHS) is supported via:
```bash
make benchmark-run TOML=configs/h1_parity_lhs.toml
```
*Note: The `benchmark_analyzer.py` statistical suite (confidence intervals, Bland-Altman plots) is currently in Beta.*

---

### 💬 Seeking Feedback & Discussion

As this is my first major project in this space, I would greatly appreciate any feedback, criticisms, or suggestions! Specifically, I'm interested in:
- **API Design & Layering:** Does the decoupled layer separation feel intuitive, or are there areas where it feels over-engineered?
- **Framework Integration & Adapters:** Suggestions or best practices for integrating external evolutionary libraries (or adding new ones).
- **Benchmarking Methodology & Tips:** Advice on refining our statistical comparison pipeline, parity evaluation suites, or standardized metric logging.
- **JAX Patterns & Optimization:** Tips on edge cases, memory optimization, or potential bottlenecks in the on-device execution loops.

Repository Link: [github.com/LeonardoDiCaterina/MalthusJAX](https://github.com/LeonardoDiCaterina/MalthusJAX)

Thank you for taking a look, and I look forward to your thoughts and feedback!
