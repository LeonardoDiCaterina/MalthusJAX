# Show and Tell: MalthusJAX — A Unified JAX Framework for Evolutionary Computation

Hi everyone! I’m sharing **MalthusJAX**, an open-source framework for hardware-accelerated Evolutionary Computation natively in JAX. 

This is my first major open-source project in this domain, and I'm sharing it here primarily to gather feedback, critiques, and suggestions from the community on the overall architecture, API design, and benchmarking methodology.

---

## 1. On-Device Execution & Scaling Behavior
At its core, MalthusJAX keeps the core evolutionary loop—from mutation and crossover to evaluation—strictly on-device using JAX transformation primitives (`jax.lax.scan`, `jax.vmap`, `jax.jit`). This eliminates host-device CPU-GPU memory transfers during multi-generation evolution runs. 

*Note on throughput:* Raw evaluation throughput alone isn't very meaningful without contextualized comparisons against other libraries (which is why we built the unified adapter and parity benchmarking suite below). The throughput numbers primarily confirm that execution is running on-device without host-device synchronization bottlenecks during scan loops:
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
To avoid per-generation conversion overhead across different state representations (population vectors, repertoire archives, graph structures), MalthusJAX unifies upstream states into a single internal representation once during initialization. External framework engines implement a unified `Engine` benchmark interface (`run_once(key) -> Dict`) returning standardized history, summary, and timing metrics.

---

## 3. Vectorized Island Models
MalthusJAX includes a vectorized implementation of island models (`BaseIslandModel`, `RingTopologyIsland`, `FullyConnectedIsland`). Instead of using Python multiprocessing or networking overhead, `jax.vmap` vectorizes local evolutionary engines across independent island sub-populations on-device. Islands evolve independently in parallel for `migration_interval` steps inside JAX, with topological migrant exchange (`jnp.roll` or global permutation) executed between migration epochs.

---

## 4. Native Multi-Objective Evolution (NSGA-II)
The library includes a native multi-objective engine (`MOEngine`). Non-dominated Pareto sorting and crowding distance calculations are natively integrated into `MOPopulation`, maintaining non-dominated ranks and crowding distances across combined parent ($\mu$) and offspring ($\lambda$) pools.

---

## 5. Focus on Parity & Fair Benchmarking
A key priority during development was ensuring fair comparisons across different algorithms:
- **Parity Verification:** We built automated test suites comparing MalthusJAX adapters side-by-side with native upstream implementations on identical random seeds.
- **Standardized Benchmark Outputs:** Our `BenchmarkRunner` wraps all adapters and native engines to return standardized `ExperimentResult` objects, capturing identical history metrics, serialization formats, and wall-clock timings across all algorithms.

**Maintenance & Version Compatibility:**
To ensure stability as upstream libraries evolve, we maintain a compatibility matrix (pinning versions such as EvoSAX 0.1.5) tested continuously via automated parity pipelines.

---

## 🧱 Layered Architecture & Technical Specifications

MalthusJAX is decoupled into distinct layers so users can interact at whichever level of abstraction fits their workflow. Each layer is backed by a technical reference specification:

1. **Layer 1: Core Primitives & State ([`malthusjax.core`](src/malthusjax/core/README.md))**
   Pure functional JAX PyTree data structures (`BasePopulation`, `BaseGenome`, `BaseEvaluator`). No hidden state—just immutable JAX arrays and dataclasses:
   - **`BaseGenome` Subclasses:** Represent genetic encodings (`RealGenome`, `BinaryGenome`, `CategoricalGenome`, `LinearGenome`) as PyTrees wrapping typed JAX arrays for parameters and domain constraints.
   - **`BasePopulation` Container:** A Struct-of-Arrays (SoA) PyTree container pairing individual genome arrays alongside population-level `fitness` vectors and static configuration.
   - **`BaseEvaluator`:** Abstract functional contracts mapping PyTree populations to evaluated fitness tensors (`SphereEvaluator`, `GriewankEvaluator`, `BoxEvaluator`, `KnapsackEvaluator`, `BBOBEvaluator`).

2. **Layer 2: Functional 3-Tier Operators ([`malthusjax.operators`](src/malthusjax/operators/README.md))**
   Stateless JAX operator dataclasses separating domain math from vectorization:
   - **3-Tier Hierarchy:** Tier 1 (`_mutate_one`, `_recombine_one`) operates on single genome PyTrees; Tier 2 (`_generate_noise`) generates noise PyTrees; Tier 3 (`__call__`) orchestrates JAX `vmap` calls over population PyTrees.
   - **Selection Operators:** `TournamentSelection` (balanced), `RouletteSelection` (softmax/Gumbel-Max), `ElitePoolSelection` ($O(N)$ `jnp.argpartition`), `EvoSaxMimicSelection`.
   - **Crossover & Mutation:** `UniformCrossover`, `BlendCrossover` (BLX-$\alpha$), `SBXCrossover`, `BitFlipMutation`, `GaussianMutation` (with schedules and bounds clipping), `PolynomialMutation`.
   - **Emitters & Injection Operators:** Quality-Diversity emitters (`BaseEmitter`, `GeneticEmitter`, `MixingEmitter`) and single-key noise injection operators (`base_injection.py`) for deterministic replay testing.

3. **Layer 3: Execution Engines ([`malthusjax.engine`](src/malthusjax/engine/README.md))**
   Hardware-accelerated execution engines written in pure JAX:
   - **Single & Multi-Objective Engines:** `GeneticEngine` (5-phase generational loop), `MOEngine` (NSGA-II paradigm), `MapElitesEngine` (QD grid search with `qdax`).
   - **Resource Budgeting (`ResourceMapper`):** Pre-calculates exact per-operator PRNG key slices during `init_state` (`KeyDerivationStrategy`: `SPLIT` vs `FOLD`), eliminating dynamic key allocation during scan execution.
   - **Vectorized Island Topologies:** `RingTopologyIsland` and `FullyConnectedIsland` running $N$ independent engines in parallel via `jax.vmap`.
   - **Scan-based Loop & Ask/Tell:** Supports `run(state)` via `jax.lax.scan` and stateful `ask(state)` / `tell(state, evaluated_pop)` APIs for custom evaluation loops.

4. **Layer 4: Composer & Adapters ([`malthusjax.composer`](src/malthusjax/composer/README.md))**
   High-level experiment orchestration layer for zero-boilerplate experiment execution:
   - **Unified Adapter Protocol:** Adapters (`build_evosax_engine`, `build_qdax_engine`, `build_tensorneat_engine`, `build_kozax_engine`) translating upstream library states into unified `Engine` benchmark objects.
   - **Catalog Registry:** String DSL lookups (`OperatorCatalog`, `EngineRegistry`, `GenomeCatalog`) parsing specs like `"tournament:num_selections=25,tournament_size=3"` or `"blend:alpha=0.5"`.
   - **Declarative TOML Integration:** Parses version-controlled TOML files via `load_experiment_config` to wire together shared defaults and per-pipeline overrides.

5. **Layer 5: Unified Benchmarking & Statistics ([`malthusjax.stats`](src/malthusjax/stats/README.md) & [`malthusjax.benchmarking`](src/malthusjax/benchmarking/README.md))**
   Standardized evaluation and statistical hypothesis testing layer:
   - **Standardized Execution Harness:** `BenchmarkRunner` executes any engine or adapter across parameterized seeds and collects structured `ExperimentResult` data.
   - **Automated Parity & Statistical Suite:** `malthusjax.stats` computes seed-aligned paired hypothesis tests (Wilcoxon signed-rank, paired $t$-test, sign test, TOST equivalence), Holm/FDR-BH multiple-testing corrections, Shapiro-Wilk normality gating, and effect sizes (Cohen's $d_z$).

---

### 💻 Code Example: Vectorized Island Model

```python
from malthusjax.engine.island_model.topologies import RingTopologyIsland
from malthusjax.engine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.real_evaluators import SphereEvaluator, SphereConfig
from malthusjax.operators.selection import TournamentSelection
from malthusjax.operators.crossover import BlendCrossover
from malthusjax.operators.mutation import GaussianMutation
import jax.random as jr

# 1. Configure base genetic engine for each island
base_engine = GeneticEngine(
    genome_config=RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0)),
    evaluator=SphereEvaluator(config=SphereConfig(maximize=False)),
    selection=TournamentSelection(num_selections=25, tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.1),
    engine_params=GeneticEngineParams(pop_size=50, num_generations=50),
)

# 2. Wrap base engine in a 4-island Ring topology model
island_model = RingTopologyIsland(
    engine=base_engine,
    num_islands=4,
    migration_interval=50,
    num_migrants=5,
)

# 3. Initialize state across islands and step (evolves islands in parallel via vmap,
# then performs topological migration between migration intervals)
key = jr.PRNGKey(42)
multi_state = island_model.init_state(key)
next_multi_state, history = island_model.step(multi_state)
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
   *Runs a MAP-Elites parity check comparing MalthusJAX's native `build_qdax_engine` directly against QDAX, monitoring QD-Score, Max Fitness, and Archive Grid Coverage.*

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

#### 🔧 Declarative TOML Configuration Schema
Experiments are declaratively specified via TOML configuration files parsed by `load_experiment_config`:

```toml
[experiment]
name = "crossover_comparison"
output_dir = "results/crossover_comparison"

[experiment.shared]
fitness       = "sphere:dim=10"
pop_size      = 50
generations   = 100
genome_length = 10
bounds        = [-5.0, 5.0]
seeds         = [42, 43, 44]
prng_impl     = "threefry2x32"
elitism       = 2
maximize      = false

[pipelines.blend_ga]
backend     = "malthusjax"
engine_type = "ga"
selection   = "tournament:num_selections=25,tournament_size=3"
crossover   = "blend:alpha=0.5"
mutation    = "gaussian:mutation_rate=0.1,mutation_strength=0.1"

[pipelines.sbx_ga]
backend     = "malthusjax"
engine_type = "ga"
selection   = "tournament:num_selections=25,tournament_size=3"
crossover   = "simulated_binary:eta=20.0"
mutation    = "polynomial:mutation_rate=0.1,eta=20.0"
```

#### 📉 Advanced: LHS Benchmarking & Statistical Analysis (Beta)
For sweeps across function spaces, Latin Hypercube Sampling (LHS) is supported via:
```bash
make benchmark-run TOML=configs/h1_parity_lhs.toml
```

---

### 💬 Seeking Feedback & Discussion

As this is my first major project in this space, I would greatly appreciate any feedback, criticisms, or suggestions! Specifically, I'm interested in:
- **API Design & Layering:** Does the decoupled layer separation feel intuitive, or are there areas where it feels over-engineered?
- **Framework Integration & Adapters:** Suggestions or best practices for integrating external evolutionary libraries (or adding new ones).
- **Benchmarking Methodology & Tips:** Advice on refining our statistical comparison pipeline, parity evaluation suites, or standardized metric logging.
- **JAX Patterns & Optimization:** Tips on edge cases, memory optimization, or potential bottlenecks in the on-device execution loops.

Repository Link: [github.com/LeonardoDiCaterina/MalthusJAX](https://github.com/LeonardoDiCaterina/MalthusJAX)

Thank you for taking a look, and I look forward to your thoughts and feedback!
