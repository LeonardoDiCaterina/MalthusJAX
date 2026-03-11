# The Engine: Orchestration and Compilation

The `malthusjax.engine` module implements Tier 2 of the framework. This tier orchestrates genomes, fitness evaluators, and genetic operators into a fully compilable evolution loop using a **scan-based stateful architecture**. The core design principle is **immutable state threading**: the evolution loop carries one opaque state object through `jax.lax.scan`, guaranteeing JAX traceability and JIT compilability to a single XLA kernel. This architecture enables three key capabilities:

1. **Deterministic reproducibility** — explicit PRNG threading with configurable key derivation strategies (SPLIT vs. FOLD)
2. **Seamless GPU/TPU deployment** — XLA buffer donation and named sharding for multi-device execution
3. **Tunable compilation surface** — separation of static configuration (compiled into the kernel) from dynamic state (traced by XLA), enabling users to control what appears in the HLO graph

The framework provides two execution modes: `step()` for iterative in-process evolution, and `ask()`/`tell()` for external fitness evaluation (e.g., remote simulations). Both modes preserve identical state semantics.

---

## 4.1. Stateful Architecture and the Scan Contract

### 4.1.1. Design Rationale: State Threading vs. Mutable Loops

JAX's functional paradigm prohibits mutable arrays and side effects within traced code. Rather than fighting this constraint, the engine embraces it by threading all intermediate data (population, best solution, generation counter, PRNG key) through a single immutable state object. At each generation, `step(state)` reads the carry, computes the next generation, and returns an updated carry. This carries two benefits: (1) **reproducibility** — given an initial state and a seed, evolution is deterministic (random keys are threaded, not drawn from an implicit global stream), and (2) **compiler transparency** — XLA sees the entire evolution loop as a pure function, enabling its optimizer to inline, fuse, and compile multiple generations into a single kernel with optimized memory reuse.

### 4.1.2. AbstractEngine — The Protocol

`AbstractEngine` defines a minimal contract: two abstract methods `init_state(seed)` and `step(state)`, plus a provided `run()` method that wraps `step` in `jax.lax.scan`. The `run()` method caches the compiled evolution kernel at the module level, keyed by the engine instance identity. This ensures each engine configuration receives a distinct XLA executable, avoiding cross-configuration cache collisions.

The engine supports two semantics for hyperparameter modification:
- **Static parameters** (e.g., population size, elitism count, PRNG strategy) are compiled into the kernel. Changing them triggers recompilation.
- **Dynamic parameters** (e.g., mutation strength, scheduled via per-generation callbacks) are traced by XLA and incur per-iteration cost but allow runtime tuning.

### 4.1.3. AbstractEvolutionState — The Scan Carry

Evolution state is decomposed into two layers: **traced fields** (population genetics and fitness values) and **metadata fields** (operator configurations, resource maps). PyTree registration controls which fields enter the JAX trace. Traced fields include population, best solution found so far, generation counter, and the PRNG key. Metadata fields—operator instances, resource allocation tables, scheduling parameters—are marked `pytree_node=False`. They ride along in the carry without bloating the JAX trace or triggering buffer allocation on each scan iteration.

This separation is configurable: users can promote metadata to traced fields for HLO transparency or demote traced fields to metadata for compilation tightness.

### 4.1.4. AbstractEngineParams — Static Configuration

Hyperparameters controlling the evolution loop are immutable dataclasses marked entirely as static configuration (`pytree_node=False`). This makes the engine hashable by instance identity, ensuring dictionary-based kernel caching is type-safe. Modification of engine parameters (e.g., changing `pop_size`) produces a new engine instance, triggering fresh XLA compilation.


## 4.2. Generational Loop Architecture: Five-Phase Orchestration

Each call to `step(state)` executes one generation in five phases: entropy allocation, selection, reproduction, population merge, and evaluation. This phase structure is architectural (not arbitrary). It separates distinct concerns—randomness management, parent selection, variation, population assembly, and fitness assessment—such that each phase can be independently personalized, traced for inspection, or optimized by the compiler.

### 4.2.1. Phase 0 — Entropy Allocation

At the start of each generation, the PRNG key is partitioned into operator-specific sub-keys using a pre-computed resource allocation table. The table specifies how many keys each operator needs and where in the flat key array they reside. This design decouples random key derivation from operator invocation: keys are allocated once, centrally, and operators receive their slices as pure inputs. This separation enables:
- **Reproducibility auditing** — key allocation is deterministic and order-independent
- **Operator composition** — operators are not responsible for key splitting; they receive ready-to-use keys
- **XLA optimization** — key allocation can be moved outside loops or fused with other arithmetic

### 4.2.2. Phase 1 — Selection

Selection identifies elite individuals for preservation and parental individuals for reproduction. The selection operator reads population fitness values and returns index arrays. Elitism is handled by concatenating elite genes with reproduced offspring; when elitism is zero, an empty structure of matching type is created to maintain tree shape. Selection is decoupled from reproduction, allowing independent tuning of selection pressure and reproduction rate.

### 4.2.3. Phase 2 — Reproduction (Crossover + Mutation)

Reproductive variation is applied in two sequential stages—crossover followed by mutation—chained as a single operation. Parent pairs are gathered from the selected indices; the crossover operator produces offspring; the mutation operator refines them. Both operators receive the generation counter, enabling time-dependent behavior (e.g., mutation rate decay) without mutating operator state. The generation counter is a tracer inside the scan loop, making schedule functions pure JAX and compatible with XLA.

### 4.2.4. Phase 3a — Merge

Elite genes and mutated offspring are combined into the next-generation population. The implementation uses JAX's `dynamic_update_slice` rather than concatenation, permitting XLA to reuse the old population buffer in-place through buffer donation. This pattern transfers physical memory ownership from one generation to the next without allocation between generations.

### 4.2.5. Phase 3b — Evaluate

The merged population is passed to the fitness evaluator, which applies the objective function via `vmap` over the population batch. Evaluation is decoupled from variation, enabling use cases where evaluation is external (ask/tell mode) or remote.

### 4.2.6. Phase 4 — Hall-of-Fame Update

The best solution across generations is tracked via configurable overhead. Three modes are available:
- **NONE**: Track the best-with-generation as a post-scan pass (zero per-step cost)
- **LIGHT**: Track monotonic improvement online (small per-step cost)
- **FULL**: Update the best genome on every generation (higher per-step cost but complete history)

Python-level branching selects which mode is compiled, ensuring only one code path enters the XLA kernel per JIT instance.

### 4.2.7. Output and Finalization

After each `step()`, generation metrics (best fitness, mean fitness, generation number) are extracted and stacked by `lax.scan` into a history array of shape `(num_generations,)`. Post-scan finalization (outside tracing) fills in missing information (e.g., best genome for NONE mode) via a one-shot pass over the final population.

---

## 4.3. Alternative Execution Mode: Ask/Tell Interface

For scenarios where fitness evaluation is external (e.g., expensive simulations, remote evaluation services, hybrid CPU/GPU pipelines), the engine provides an `ask()`/`tell()` interface instead of the direct `step()` method. The call `engine.ask(state)` allocates entropy and returns a population candidate for external evaluation (alongside a buffered key receipt). The call `engine.tell(state, evaluated_pop)` consumes the buffered keys, executes selection through evaluation, and returns the updated state. This pattern maintains identical state semantics and allows integration with non-JAX fitness functions while preserving reproducibility through explicit key threading.

---

## 4.4. Deterministic Resource Allocation

### 4.4.1. Resource Maps

At initialization, the engine computes a resource allocation plan specifying how many PRNG keys each operator needs and where in the flat key array they are located. This pre-computation enables zero-overhead key slicing during each generation: the key derivation is done once, and operators receive their slices as pure inputs without dynamic lookup or recursion. The resource map is marked metadata (not traced), avoiding JAX overhead.

### 4.4.2. Key Derivation Strategies

Two strategies are available for generating the per-generation key block:
- **SPLIT**: Sequential hash chain (`jax.random.split`), lower memory but sequential parallelism
- **FOLD**: Fully parallel (`vmap(fold_in)`), higher memory overhead but deterministic and seekable

The choice is static (compiled into the kernel) and specified at engine creation time. The FOLD strategy is incompatible with certain PRNG implementations (e.g., RBG) where `fold_in` is unsupported.

### 4.4.3. Sharding and Multi-Device Distribution

For multi-device execution (e.g., data-parallel evolution across GPU devices), the engine provides a `ShardingManager` that enforces JAX `NamedSharding` specifications. This component is optional and non-intrusive—it does not affect the evolution semantics on single devices but enables deployment across a device mesh when instantiated. Population buffers are allocated with explicit sharding metadata, allowing XLA to optimize communication patterns and buffer placement.

---

## 4.5. Operator-Level Personalization and Scheduling

### 4.5.1. Generation-Aware Operators

Operators accept an optional generation counter as a parameter, enabling time-dependent behavior (e.g., mutation rate decay, crossover bias modulation) without mutating operator state. Scheduling functions are pure JAX, compiled into the kernel, and safe inside `jax.lax.scan`. This design decouples scheduling logic from operator logic, allowing users to implement custom schedules (e.g., Bayesian optimization of decay rates) without forking operators.

Schedule types include:
- **CONSTANT**: Time-independent (baseline)
- **LINEAR_DECAY**: Strength decreases linearly from initial to final value
- **COSINE_ANNEAL**: Cosine annealing schedule
- **EXPONENTIAL_DECAY**: Exponential decay schedule

Scheduling parameters are static configuration; changing them triggers recompilation.

### 4.5.2. Configurable Hall-of-Fame Tracking

Post-evaluation, the engine updates its best-solution record. Three modes control the overhead and completeness of this tracking:
- **NONE** (zero cost, post-hoc recovery): Best solution is found after evolution completes
- **LIGHT** (minimal cost, monotonic): Best fitness is tracked but not genome
- **FULL** (per-generation cost, complete history): Both best fitness and genome are updated every step

The selected mode is baked into the compiled kernel via Python branching outside the JIT boundary.

---

## 4.6. Compilation and HLO Transparency

### 4.6.1. Scan-Based Kernel Caching

The evolution loop is compiled once per engine instance. The cached kernel is retrieved by `id(engine)`, ensuring each configuration has a distinct XLA executable. After compilation, XLA optimizes the entire evolution process as a single function, applying loop fusion, buffer reuse, and kernel specialization. The unroll depth is automatic (XLA decides); manual configuration is deprecated.

### 4.6.2. HLO Tracing and Named Call Labels

Each evolution phase can be wrapped with HLO labels for profiling and inspection. By default, labels are disabled, allowing full fusion. When enabled (e.g., `GeneticEngineParams.debug_tracing=True`), `jax.named_call` annotations insert readable phase names into the HLO graph, facilitating performance debugging. Enabling tracing incurs no overhead when disabled (the branch is resolved before XLA sees the code).

Users can extract the compiled HLO text via `engine.get_hlo_text(state)` for inspection and custom optimization.

### 4.6.3. Buffer Donation and Memory Reuse

The merge phase writes elite and mutant genes into the old population buffer, enabling XLA to donate (reuse) that buffer rather than allocate fresh memory. This pattern cascades through generations: each generation's population buffer is the destination for the next, reducing memory traffic and improving cache locality.


## 4.7. Initialization and State Management

`init_state(seed)` is a one-time setup phase that:
1. Initializes the first population via the genome's `random_init()` method
2. Evaluates the initial population to seed the fitness values
3. Configures all operators (setting input dimensions, scheduling bounds, key typing)
4. Computes the resource allocation table
5. Returns a fully ready `GeneticEvolutionState`

The initialized state is immutable and can be threaded through repeated `step()` calls (or parallel `run()` calls), enabling batch experiments with identical starting conditions. After calling `engine.run(state)`, the state's buffers are consumed; for repeated runs, reinitialize via `init_state(seed)` with a fresh seed.

---

## 4.8. Worked Example: Customizing an Engine

The following example, derived directly from the engine's test suite, demonstrates how the customization axes described in this chapter are composed in practice. A minimal engine targeting the Sphere benchmark is constructed with: (1) cosine annealing on mutation strength, (2) the FOLD key derivation strategy for fully parallel entropy allocation, and (3) LIGHT hall-of-fame tracking for minimal overhead with monotonic fitness history.

```python
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.resource_mapper import KeyDerivationStrategy
from malthusjax.engine.schedules import ScheduleType, TrackBest
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection

# --- Static configuration (compiled into the kernel) ---
engine_params = GeneticEngineParams(
    pop_size=50,
    elitism=3,
    num_generations=200,
    key_derivation=KeyDerivationStrategy.FOLD,   # parallel entropy allocation
    schedule_type=ScheduleType.COSINE_ANNEAL,    # mutation strength schedule
    initial_strength=0.5,
    final_strength=0.01,
    track_best=TrackBest.LIGHT,                  # monotonic history, low overhead
)

genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
evaluator = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=10, maximize=False))

engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=genome_config,
    evaluator=evaluator,
    selection=ElitePoolSelection(num_selections=50, elite_k=3),
    crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
    mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
)

# --- Run: state is initialized, evolution loop JIT-compiled and executed ---
state = engine.init_state(jar.PRNGKey(42))
final_state, history, timings = engine.run(state)

# history.best_fitness is monotonically non-decreasing (TrackBest.LIGHT)
# Recover stagnation post-hoc without any per-step overhead:
import jax.numpy as jnp
stagnation_mask = jnp.diff(history.best_fitness) == 0
```

Each engine parameter is a distinct customization axis: swapping `ScheduleType.COSINE_ANNEAL` for `ScheduleType.CONSTANT` changes only the schedule branch compiled into the kernel; swapping `KeyDerivationStrategy.FOLD` for `SPLIT` changes only the entropy allocation pattern; switching `TrackBest.LIGHT` for `TrackBest.FULL` changes only the hall-of-fame update path. All other phases remain unchanged, reflecting the phase-separation design of §4.2.

