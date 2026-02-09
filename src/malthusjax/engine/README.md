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

```python
p1_pop = population[parent_indices[:num_pairs]]
p2_pop = population[parent_indices[num_pairs:]]
offspring_pop = operators.crossover(k_cross, p1_pop, p2_pop, config)
final_pop = operators.mutation(k_mut, offspring_pop, config)
```

#### _merge()

Combines elites and top mutants to create next population of size `pop_size`.

```python
num_elites = elites_genes.shape[0]
num_mutants = pop_size - num_elites
next_genes = concatenate([elites_genes, mutants_genes[:num_mutants]])
```

#### _evaluate()

Computes fitness on merged population using `BaseEvaluator.evaluate_population()`.

#### _update_hof()

Tracks best genome and increments stagnation counter. Stagnation resets when a new best is found.

---

## 4) Resource Mapping

### The ResourceMap Contract

`ResourceMap` precomputes:

1. **Total RNG Budget**: Sum of keys needed across all operator stages.
2. **Per-Operator Slices**: Start/end indices for key allocation.
3. **Data Counts**: Input/output population sizes at each stage.

Example (pop_size=100, elitism=5, uniform 2-offspring crossover):

```
Selection:     input=100  →  output=100 (parent indices)        [keys allocated]
Crossover:     input=100  →  output=200 (2 offspring × pairs)   [keys allocated]
Mutation:      input=200  →  output=200 (mutants)               [keys allocated]
Next-key:                 →  output=1 (for next generation)     [1 key]
```

### Key Derivation Strategies

`KeyDerivationStrategy` enum controls RNG key generation in `ResourceMap.get_keys()`:

| Strategy | Method | Trade-off |
|----------|--------|-----------|
| **SPLIT** | Sequential `jax.random.split()` | Uncorrelated keys; sequential bottleneck |
| **FOLD** | Parallel `jax.random.fold_in()` | Deterministic; better for parallelism |

**Usage**:

```python
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    elitism=5,
    key_derivation=KeyDerivationStrategy.FOLD
)

**PRNG Implementation Control**: If you want explicit control over the PRNG backend (for example, favoring Philox on GPUs), create your master key using `malthusjax.core.random.create_key(seed, impl=PRNGImpl.PHILOX)` and pass it to `engine.init_state()`. The engine also accepts an integer seed directly (it will create a key using `engine.engine_params.prng_impl`). Passing a legacy `jax.random.PRNGKey` will still work but emits a `DeprecationWarning`.```

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

**Configuration**:

```python
def schedule(generation: int) -> float:
    """Linearly decay mutation strength from 0.5 to 0.01."""
    return 0.5 * (1.0 - generation / num_generations)

engine_params = GeneticEngineParams(
    ...,
    mutation_strength_schedule=schedule
)
```

The schedule function is called each generation in `_get_active_operators()` and applied to the mutation operator before reproduction.

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
