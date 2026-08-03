# malthusjax.engine — Technical Specification & Architecture ✅

This document describes the design, execution model, and extension patterns for the `malthusjax.engine` package. It targets developers who will implement evolutionary algorithms, tune engine behavior, or integrate custom evolution strategies into MalthusJAX.

---

## 1) The Evolutionary Engine Paradigm

### Core Idea

The engine layer orchestrates the complete evolutionary loop: selection → reproduction (crossover + mutation) → evaluation → best-genome tracking. MalthusJAX provides an abstract engine interface (`AbstractEngine`) and a concrete implementation (`GeneticEngine`) optimized for JAX/XLA.

Key principles:

- **Composition**: Genomes/fitness (Level 1) and operators (Level 2) are composed into engines (Level 3) without modification.
- **Resource Budgeting**: RNG requirements are precomputed at initialization via `ResourceMap`, enabling static allocation of key budgets across operators.
- **Init-Phase Compilation**: Operators are frozen with static input sizes at `init_state()`, allowing XLA to compile kernels once and reuse them across generations.
- **Immutability**: All engine state uses `@struct.dataclass` for traceability and safe JAX operations.

---

## 2) The Three Architectural Layers

### Layer 1: Core Components (Genomes, Fitness)

- **Genomes** (`core/genome/`): Immutable, JAX-compatible representations of candidate solutions.
- **Fitness Evaluators** (`core/fitness/`): Pure functions that compute objective values for populations.

These are **input**-level abstractions and remain unchanged by the engine.

### Layer 2: Genetic Operators

- **Selection** (`operators/selection/`): Choose candidates based on fitness (e.g., tournament, roulette).
- **Crossover** (`operators/crossover/`): Recombine selected parents (e.g., single-point, uniform).
- **Mutation** (`operators/mutation/`): Introduce variability (e.g., bit-flip, Gaussian).

These are **building blocks** for the engine and must follow strict, JAX-native signatures.

### Layer 3: Evolution Engines

- **AbstractEngine**: Interface defining the evolution contract (`init_state`, `step`).
- **GeneticEngine**: High-performance GA implementation with init-phase compilation and resource budgeting.

---

## 3) The GeneticEngine Execution Model

### Initialization Phase: `init_state(rng_key)`

1. **Compute ResourceMap**:
   - Calls `compute_resource_map()` to calculate exact RNG budget for selection, crossover, mutation, and next-generation key.
   - Validates operator input/output counts.

2. **Freeze Operators**:
   - Calls `set_input_length()` on selection, crossover, and mutation operators.
   - This locks operator parameters (e.g., population size for selection) so XLA compiles a single kernel per operator.

3. **Initialize Population**:
   - Uses genome class's `init_random()` to create initial population.
   - Evaluates initial fitness with the provided `BaseEvaluator`.

4. **Create GeneticEvolutionState**:
   - Bundles population, best genome, generation counter, RNG key, `ResourceMap`, and frozen `OperatorState`.
   - This state carries the complete plan for all subsequent steps.

**Result**: Static setup happens once. `step()` reuses compiled kernels across all generations.

---

### Evolution Step: `step(state)`

Each generation executes these phases (methods decorated with `@traceable` for HLO profiling):

```
1. _allocate_entropy()      — Derive RNG keys for selection/crossover/mutation
2. _get_active_operators()  — Apply mutation schedule if configured
3. _selection_phase()       — Elite extraction + parent selection
4. _reproduction_phase()    — Crossover + mutation on selected parents
5. _merge()                 — Combine elites + top mutants to fill pop_size
6. _evaluate()              — Compute fitness on merged population
7. _update_hof()            — Track best genome and stagnation counter
```

#### _allocate_entropy()

Slices pre-derived RNG keys for each operator stage via `ResourceMap.get_keys()`.

```python
all_keys = rmap.get_keys(state.rng_key)
k_sel_slice = all_keys[rmap.get_key_slice('selection')]
k_cross = all_keys[rmap.get_key_slice('crossover')]
k_mut = all_keys[rmap.get_key_slice('mutation')]
k_next = all_keys[rmap.get_key_slice('next_key')][0]
```

**Key derivation** is controlled by `GeneticEngineParams.key_derivation`:
- `SPLIT`: Sequential `jax.random.split()` — uncorrelated but single-threaded.
- `FOLD`: Parallel `jax.random.fold_in()` — deterministic but more parallelizable.

#### _selection_phase()

Extracts elites (top `elitism` individuals) and selects parents via the configured selection operator.

```python
_, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
elites_genes = population[elite_idx].genes
selected_idx = operators.selection(key_selection, population)
```

#### _reproduction_phase()

Crosses selected parents and mutates offspring. Operator input sizes were frozen at `init_state()` via `set_input_length()`.

Genes are gathered directly (not full populations) to avoid copying fitness arrays that crossover never reads (FB-5). Lightweight parent populations are built with `spawn_offspring(genes, fitness=jnp.zeros(n))`, skipping the default NaN allocation (FB-2).

```python
# FB-5: Gather genes only — crossover never reads fitness
p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
dummy_fitness = jnp.zeros(num_pairs)
p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

offspring_pop = operators.crossover(k_cross, p1_pop, p2_pop, config)
final_pop = operators.mutation(k_mut, offspring_pop, config)
```

#### _merge()

Combines elites and top mutants to create next population of size `pop_size`. Uses `jax.lax.dynamic_update_slice` instead of `jnp.concatenate` to enable XLA buffer donation (FB-3).

```python
num_elites = elites_genes.shape[0]
num_mutants = pop_size - num_elites
# Write elites then mutants into a pre-allocated buffer (dynamic_update_slice)
```

#### _evaluate()

Computes fitness on merged population using `BaseEvaluator.evaluate_population()`.

#### _update_hof()

Tracks best genome via `TrackBest` enum. Uses `jnp.where` (no `jax.lax.cond` fusion barrier) to conditionally update the best genome (FB-4).

---

## 4) Resource Mapping — Advanced Guide

### The ResourceMap Concept

**ResourceMap** is the engine's master plan: a pre-computed blueprint that specifies:
1. **Total RNG budget**: Exact number of keys needed for all operators in one generation
2. **Per-operator allocations**: Start/end indices for each operator's key slice
3. **Data flow counts**: Input/output sizes at each pipeline stage (selection → crossover → mutation)
4. **Operator specifications**: Frozen input lengths enabling static XLA compilation

**Why pre-computation?**
- Enables static computation of XLA kernels (compiled once, reused every generation)
- Avoids runtime conditionals and shape inference
- Catches configuration errors early (before JIT)
- Allows precise PRNG key allocation without runtime re-sizing

### ResourceMap Structure

```
ResourceMap fields:
├── total_rng_budget: int
│   └── Total keys needed for all operators + next-generation key
├── selection: OperatorAllocation
│   ├── num_keys: int (keys needed for selection)
│   ├── start_idx, end_idx: int (slice into key buffer)
│   ├── input_count: int (population size)
│   └── output_count: int (number of parents selected)
├── crossover: OperatorAllocation
│   ├── num_keys: int
│   ├── input_count: int (number of parents)
│   └── output_count: int (number of offspring produced)
├── mutation: OperatorAllocation
│   ├── num_keys: int
│   ├── input_count: int (offspring count from crossover)
│   └── output_count: int (mutated offspring count)
├── next_key: OperatorAllocation
│   ├── num_keys: 1 (always 1 key for next generation)
│   └── (used to seed next generation's master key)
├── pop_size: int (population size, baked into map)
├── num_pairs: int (number of parent pairs for crossover)
├── genome_shape: Tuple[int, ...]
└── key_derivation: KeyDerivationStrategy (SPLIT or FOLD)
```

### Data Flow Example

For **pop_size=100, elitism=5, num_offspring=2**:

```
┌─ SELECTION ────────────────┐
│  Input:  100 (population)  │
│  Output: 100 (parent_idx)  │  ← Selection doesn't change count
│  Keys:   N (depends on tournament_size, etc.)
└────────────────────────────┘
                ↓
┌─ CROSSOVER ────────────────────────────────┐
│  Input:  100 (parents)                     │
│  Pairs:  50 (100 parents / 2 per pair)     │
│  Output: 100 (50 pairs × 2 offspring)      │  ← Balanced for pop_size
│  Keys:   M (depends on crossover type)     │
└────────────────────────────────────────────┘
                ↓
┌─ MUTATION ──────────────────────────────────┐
│  Input:  100 (offspring)                    │
│  Output: 100 (mutated offspring)            │  ← 1:1 mapping by default
│  Keys:   K (depends on mutation_rate)       │
└─────────────────────────────────────────────┘
                ↓
         [Next-Gen Key: 1]
```

### Computing Resource Maps in Practice

```python
from malthusjax.engine.resource_mapper import compute_resource_map

# Create operators
selection = TournamentSelection(num_selections=100, tournament_size=3)
crossover = UniformCrossover(crossover_rate=0.8)
mutation = GaussianMutation(mutation_rate=0.1, mutation_strength=0.2)

# Compute the resource map
rmap = compute_resource_map(
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    genome_config=RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0)),
    pop_size=100,
    key_derivation=KeyDerivationStrategy.SPLIT,
)

# Inspect the map
print(f"Total RNG budget: {rmap.total_rng_budget}")
print(f"Selection keys:   {rmap.selection.num_keys} (slice {rmap.selection.start_idx}:{rmap.selection.end_idx})")
print(f"Crossover keys:   {rmap.crossover.num_keys} (slice {rmap.crossover.start_idx}:{rmap.crossover.end_idx})")
print(f"Mutation keys:    {rmap.mutation.num_keys} (slice {rmap.mutation.start_idx}:{rmap.mutation.end_idx})")
print(f"Next-key:         {rmap.next_key.num_keys} (slice {rmap.next_key.start_idx}:{rmap.next_key.end_idx})")

# Data flow
print(f"\nData Flow:")
print(f"  Selection: {rmap.selection.input_count} → {rmap.selection.output_count}")
print(f"  Crossover: {rmap.crossover.input_count} → {rmap.crossover.output_count} ({rmap.num_pairs} pairs)")
print(f"  Mutation:  {rmap.mutation.input_count} → {rmap.mutation.output_count}")
```

### Extracting Keys from Master Key

Once a `ResourceMap` is computed, `engine.step()` uses it to allocate keys:

```python
# Inside GeneticEngine.step() — entropy allocation phase
all_keys = rmap.get_keys(master_key)  # Derives total_rng_budget keys

# Extract slices for each operator
k_sel = all_keys[rmap.get_key_slice("selection")]      # Keys for selection
k_cross = all_keys[rmap.get_key_slice("crossover")]    # Keys for crossover
k_mut = all_keys[rmap.get_key_slice("mutation")]       # Keys for mutation
k_next = all_keys[rmap.get_key_slice("next_key")][0]   # Next-gen key
```

### Key Derivation Strategies in Depth

#### SPLIT Strategy (Default)

```
master_key → jax.random.split(master_key, N) → [key_0, key_1, ..., key_N-1]

Characteristics:
✓ Maximally uncorrelated keys (gold standard)
✓ Traditional approach, well-tested
✗ Sequential bottleneck (cannot be parallelized)
✗ All N-1 split operations run sequentially
→ Use when: Multi-device or reproducibility is critical
```

**Example**:
```python
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    elitism=5,
    key_derivation=KeyDerivationStrategy.SPLIT,
)
```

#### FOLD Strategy (Parallelizable)

```
master_key + indices [0, 1, ..., N-1]
    → vmap(jax.random.fold_in(master_key, i))
    → [key_0, key_1, ..., key_N-1]

Characteristics:
✓ Fully parallelizable (jax.vmap can vectorize)
✓ Better for large key budgets
✓ Deterministic (same seed always gives same keys)
✗ Not all PRNG backends support fold_in (e.g., RBG/UNSAFE_RBG)
✗ Slightly higher computational cost per key
→ Use when: Single-device GPU/TPU with large key budgets
```

**Example**:
```python
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    elitism=5,
    key_derivation=KeyDerivationStrategy.FOLD,
)
```

**Compatibility Check**:
```python
import jax
from malthusjax.engine.resource_mapper import KeyDerivationStrategy

# Check if FOLD is compatible with your PRNG backend
prng_impl = getattr(jax.config, "jax_default_prng_impl", "threefry2x32")
print(f"Current PRNG impl: {prng_impl}")

if prng_impl in ("rbg", "unsafe_rbg"):
    print("⚠️  FOLD strategy NOT compatible. Use SPLIT.")
else:
    print("✓ FOLD strategy is compatible.")
```

### PRNG Implementation Control

If you want explicit control over the PRNG backend (e.g., favoring Philox on GPUs):

```python
from malthusjax.core.random import create_key, PRNGImpl

# Use Philox PRNG (fast on GPU, may vary cross-platform)
key = create_key(seed=42, impl=PRNGImpl.PHILOX)

# Or use Threefry (deterministic on all platforms, default)
key = create_key(seed=42, impl=PRNGImpl.THREEFRY)

# Pass to engine
state = engine.init_state(key)
```

**Backward compatibility**: Passing a legacy `jax.random.PRNGKey` still works but emits a `DeprecationWarning`.

### Inspecting Resource Maps

Use `get_resource_summary()` for a cascade view:

```python
from malthusjax.engine.resource_mapper import get_resource_summary, compute_resource_map

rmap = compute_resource_map(...)
print(get_resource_summary(rmap))
```

**Output**:
```
Pipeline Resource & Flow Summary:
  Total RNG Budget: 45 keys

  [1. SELECTION]
     In: 100 (Pop Size) -> Out: 100 indices (Parents needed)
     Keys: 10 (Slice 0:10)

  [2. CROSSOVER]
     In: 100 parents (50 pairs) -> Out: 100 offspring
     Keys: 15 (Slice 10:25)

  [3. MUTATION]
     In: 100 offspring -> Out: 100 mutants
     Keys: 19 (Slice 25:44)

  [4. NEXT-KEY]
     Keys: 1 (Slice 44:45)
```

### Advanced: ShardingManager

For multi-device execution, `ShardingManager` optimizes tensor layout using GSPMD:

```python
from malthusjax.engine.resource_mapper import ShardingManager

# Create sharding manager with axis "batch" for data parallelism
shard_mgr = ShardingManager(axis_name="batch")

# Allocate sharded population array
pop_shape = (100, 10)  # 100 individuals, 10-D genomes
pop_array = shard_mgr.alloc_population(pop_shape, dtype=jnp.float32)
# Tensor placed on device with batch-major sharding

# Allocate sharded fitness vector
fitness_shape = (100,)
fitness_sharding = shard_mgr.vector_sharding
fitness_array = jax.device_put(jnp.zeros(fitness_shape), fitness_sharding)

# Replicate metadata (best_fitness, scalars)
best_fitness = jax.device_put(0.0, shard_mgr.replicated_sharding)
```

---

## 5) Operator Freezing with set_input_length()

Operators have a static input size (e.g., population size for selection, number of parent pairs for crossover) that is known at initialization. The `set_input_length()` method freezes this size:

1. **Enables operator-specific optimizations**: Selection can pre-allocate tournament pools; crossover can pre-size buffers.
2. **Stabilizes XLA compilation**: One kernel per operator, reused every generation.
3. **Simplifies the step loop**: No need to pass input length to every operator call.

**Example**:

```python
# In init_state()
active_sel = self.selection \
    .set_input_length(rmap.selection.input_count)
    # ^ frozen at initialization; reused every generation

# In step()
selected_idx = active_sel(key_selection, population)
# ^ uses pre-frozen input_length, no recompilation
```

---

## 6) Mutation Strength Scheduling

Engines can optionally apply a time-dependent mutation strength schedule to vary exploration over generations.

**Configuration (new — JAX-native)**:

```python
from malthusjax.engine.schedules import ScheduleType

engine_params = GeneticEngineParams(
    ...,
    schedule_type=ScheduleType.LINEAR_DECAY,
    initial_strength=0.5,
    final_strength=0.01,
)
```

Available schedules: `CONSTANT` (default), `LINEAR_DECAY`, `COSINE_ANNEAL`, `EXPONENTIAL_DECAY`.

All schedule computation uses `jnp` operations, so it is safe inside `jax.lax.scan` — no recompilation per generation.

> **Deprecated**: The `mutation_strength_schedule` callable parameter is still accepted for backward compatibility but will be removed in v0.4.0. Prefer the `schedule_type` / `initial_strength` / `final_strength` fields.

---

## 7) Ask/Tell Interface

For decoupled evaluation workflows (e.g., external simulators, distributed evaluation), the engine provides an ask/tell interface:

```python
# Ask: get the next population to evaluate
engine_with_entropy, population = engine.ask(state)

# ... evaluate externally ...
new_population = external_evaluator(population)

# Tell: update state with evaluated population
next_state = engine.tell(state, new_population)
```

**Implementation**:
- `ask()` allocates entropy and returns the population to evaluate externally.
- `tell()` completes the evolutionary step using the pre-evaluated population.

---

## 8) Sharding and Device Placement

`ShardingManager` provides GSPMD (General and Simplified Parallelization) sharding layouts for population arrays:

**For single-device**: Optimizes memory layout and L-cache alignment.

**For multi-device**: Ensures proper data distribution across shards:

```python
sharding_mgr = ShardingManager(axis_name='batch')

# Population matrices (batch-major)
matrix_sharding = NamedSharding(mesh, P('batch', None))

# Fitness vectors
vector_sharding = NamedSharding(mesh, P('batch'))

# Replicated scalars (best_fitness, etc.)
replicated_sharding = NamedSharding(mesh, P())
```

---

## 9) Custom Engine Implementations

Developers can extend `AbstractEngine` to implement:

1. **Custom phases**: Override selection, reproduction, or evaluation logic.
2. **Custom metrics**: Extend `AbstractGenerationOutput` to track domain-specific KPIs.
3. **Custom state**: Subclass `AbstractEvolutionState` to track additional metadata (e.g., diversity, lineage).

**Example**:

```python
@struct.dataclass
class IslandEvolutionState(AbstractEvolutionState[...]):
    """State for island-model GA."""
    islands: Tuple[BasePopulation[...], ...] = struct.field()
    migration_history: chex.Array = struct.field()

class IslandEngine(AbstractEngine[...]):
    def step(self, state: IslandEvolutionState) -> Tuple[IslandEvolutionState, AbstractGenerationOutput]:
        # Implement island-specific logic
        ...
```

---

## 10) Best Practices

### General

- **Use immutable state**: Always use `state.replace(...)` to update state.
- **Avoid Python control flow in JIT**: Keep loops and conditionals at Python level, outside traced regions.
- **Profile with JAX tools**: Use `jax.named_call` (already in `GeneticEngine`) and JAX's profiler for HLO inspection.

### Resource Management

- **Precompute ResourceMap**: Called once in `init_state()`; reused in `step()`.
- **Validate RNG budget**: Operators should respect their allocated key slices.
- **Choose key derivation carefully**:
  - Use `SPLIT` for multi-device to ensure independent streams per device.
  - Use `FOLD` when reproducibility is critical.

### Sharding

- **Single-device**: Let `ShardingManager` optimize memory layout; overhead is minimal.
- **Multi-device**: Ensure batch axis aligns with device mesh dimensions.
- **Replicate scalars**: Non-batched values (best_fitness, generation) should use replicated sharding.

### Mutation Scheduling

- **Define schedules as pure functions**: `schedule(generation: int) -> float`.
- **Test schedule output**: Ensure returned values are valid and in expected ranges.
- **Combine with elitism**: Scheduled mutation works best with elite preservation.

---

## 10.5) Quick Examples

### Basic Engine Setup and Evolution

```python
import jax
import jax.numpy as jnp
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.core.fitness.real_evaluators import SphereEvaluator, SphereConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.selection import TournamentSelection
from malthusjax.operators.crossover import UniformCrossover
from malthusjax.operators.mutation import GaussianMutation

# Setup: Minimize sphere function f(x) = sum(x^2)
config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
evaluator = SphereEvaluator(config=SphereConfig(maximize=False))

# Create engine configuration
engine_params = GeneticEngineParams(
    pop_size=50,
    num_generations=100,
    elitism=5,
)

# Build engine with operators
engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=config,
    evaluator=evaluator,
    selection=TournamentSelection(num_selections=50, tournament_size=3),
    crossover=UniformCrossover(crossover_rate=0.8),
    mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.2),
)

# Initialize state and run evolution
key = jax.random.PRNGKey(42)
state = engine.init_state(key)

# Run full evolution loop (100 generations)
# For accessing results immediately, convert to Python scalars/arrays before return
final_state, history, info = engine.run(state, compile=True)

# Results can be accessed immediatly after run():
# print(f"Best fitness: {final_state.best_fitness}")
# print(f"Best genome: {final_state.best_genome.values}")
```

### Tracking Fitness History

```python
# Manually iterate generations to track history
key = jax.random.PRNGKey(42)
state = engine.init_state(key)

best_fitness_per_gen = []
diversity_per_gen = []

for gen in range(100):
    state, generation_output = engine.step(state)
    
    # Immediately convert to Python scalars to avoid array deletion issues
    best = float(generation_output.best_fitness)
    best_fitness_per_gen.append(best)
    
    # Optional: compute diversity metric
    fitness_array = jnp.array(state.population.fitness)  # Coerce to array
    fitness_range = float(jnp.max(fitness_array) - jnp.min(fitness_array))
    diversity_per_gen.append(fitness_range)
    
    if gen % 20 == 0:
        print(f"Gen {gen}: best={best:.6f}, diversity={fitness_range:.6f}")

# Analyze convergence (now safe - data is in Python lists)
import matplotlib.pyplot as plt
plt.plot(best_fitness_per_gen, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
```

### Using Mutation Schedules

```python
from malthusjax.engine.schedules import ScheduleType

# Setup engine with LINEAR_DECAY mutation schedule
engine_params = GeneticEngineParams(
    pop_size=50,
    num_generations=200,
    elitism=5,
    schedule_type=ScheduleType.LINEAR_DECAY,    # High → Low exploration
    initial_strength=0.5,                         # Exploration early
    final_strength=0.01,                          # Exploitation late
)

engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=config,
    evaluator=evaluator,
    selection=TournamentSelection(num_selections=50, tournament_size=3),
    crossover=UniformCrossover(crossover_rate=0.8),
    mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.2),
)

# Run with schedule: mutation strength will decay linearly across generations
state = engine.init_state(key)
final_state, history, info = engine.run(state, compile=True)

# Alternative: COSINE_ANNEAL for smoother exponential decay
engine_params_cosine = GeneticEngineParams(
    pop_size=50,
    num_generations=200,
    elitism=5,
    schedule_type=ScheduleType.COSINE_ANNEAL,
    initial_strength=0.5,
    final_strength=0.01,
)
```

### Ask/Tell Interface (Decoupled Evaluation)

```python
# Useful for external evaluation (simulations, external systems)
key = jax.random.PRNGKey(42)
state = engine.init_state(key)

for gen in range(50):
    # Ask: get entropy and population structure for external evaluation
    state, population_to_eval = engine.ask(state)
    
    # Evaluate externally (e.g., call external simulator, distributed system)
    # This could be CPU-bound, distributed, or on a different device
    fitness_scores = external_simulator(population_to_eval.genes.values)
    
    # Tell: update state with evaluated population
    evaluated_pop = population_to_eval.replace(fitness=fitness_scores)
    state, generation_output = engine.tell(state, evaluated_pop)
    
    if gen % 10 == 0:
        best = float(generation_output.best_fitness)  # Convert to Python scalar
        print(f"Gen {gen}: best={best:.6f}")
```

### Key Derivation Strategies

```python
from malthusjax.engine.resource_mapper import KeyDerivationStrategy

# Strategy 1: SPLIT (better for multi-device reproducibility)
engine_params_split = GeneticEngineParams(
    pop_size=50,
    num_generations=100,
    elitism=5,
    key_derivation=KeyDerivationStrategy.SPLIT,
)

# Strategy 2: FOLD (better parallelization on single-device)
engine_params_fold = GeneticEngineParams(
    pop_size=50,
    num_generations=100,
    elitism=5,
    key_derivation=KeyDerivationStrategy.FOLD,
)

engine_fold = GeneticEngine(
    engine_params=engine_params_fold,
    genome_config=config,
    evaluator=evaluator,
    selection=TournamentSelection(num_selections=50, tournament_size=3),
    crossover=UniformCrossover(crossover_rate=0.8),
    mutation=GaussianMutation(mutation_rate=0.1, mutation_strength=0.2),
)

# Both produce valid results; FOLD may have higher throughput on single-device
state = engine_fold.init_state(key)
final_state, history, info = engine_fold.run(state, compile=True)
```

### Accessing Generation History

```python
# Engine.run() returns history of all generations
state = engine.init_state(key)
final_state, history, info = engine.run(state, compile=True)

# Convert history to Python scalars for analysis
best_fitness_history = [float(h.best_fitness) for h in history]

# Access per-generation statistics
print(f"Number of generations: {len(best_fitness_history)}")
print(f"Final best fitness: {best_fitness_history[-1]:.6f}")
print(f"Improvement: {best_fitness_history[0] - best_fitness_history[-1]:.6f}")

# Plot convergence curve
import matplotlib.pyplot as plt
plt.plot(best_fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Convergence Over Time')
plt.show()

# Get info dict
print(f"Info keys: {info.keys()}")  # e.g., compile_time, execution_time
```

---

## 11) File Organization

```
src/malthusjax/engine/
├── __init__.py                 # Public API
├── base.py                     # Abstract base classes (AbstractEngine, AbstractEngineParams, AbstractEvolutionState)
├── genetic_fastengine.py       # Concrete GeneticEngine implementation
├── resource_mapper.py          # ResourceMap, KeyDerivationStrategy, ShardingManager
└── README.md                   # This file
```

---

## 12) Summary Table: Engine Components

| Component | Purpose | Type | Immutable |
|-----------|---------|------|-----------|
| `AbstractEngine` | Interface for evolution algorithms | ABC | N/A |
| `GeneticEngine` | Standard GA implementation | `@struct.dataclass` | ✅ |
| `GeneticEngineParams` | Configuration (pop_size, elitism, schedule, etc.) | `@struct.dataclass` | ✅ |
| `GeneticEvolutionState` | State bundle (population, best genome, RNG, ResourceMap) | `@struct.dataclass` | ✅ |
| `ResourceMap` | RNG budget and per-operator allocation details | `@struct.dataclass` (no pytree) | ✅ |
| `KeyDerivationStrategy` | SPLIT vs FOLD RNG derivation | Enum | N/A |
| `ShardingManager` | GSPMD sharding layout management | Python class | ✅ (stateless) |
| `OperatorState` | Frozen operators with static input sizes | `@struct.dataclass` | ✅ |

---

## 13) References & Related Documentation

- [Level 1: Genomes](../core/genome/README.md)
- [Level 1: Fitness](../core/fitness/README.md)
- [Level 2: Selection](../../operators/selection/README.md)
- [Level 2: Crossover](../../operators/crossover/README.md)
- [Level 2: Mutation](../../operators/mutation/README.md)
- [Main README](../../README.md)

---

**Status**: ✅ Complete and Production-Ready

This design prioritizes **composability**, **reproducibility**, and **performance** on modern JAX/XLA backends. The architecture scales from single-device laptops to multi-GPU/TPU clusters without code changes.
