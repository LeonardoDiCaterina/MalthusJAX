# malthusjax.engine — Technical Specification & Architecture ✅

This document describes the design, execution model, and extension patterns for the `malthusjax.engine` package. It targets developers who will implement evolutionary algorithms, tune engine behavior, or integrate custom evolution strategies into MalthusJAX.

---

## 1) The Evolutionary Engine Paradigm

### Core Idea

The engine layer orchestrates the complete evolutionary loop: selection → reproduction (crossover + mutation) → evaluation → hall of fame (HOF) update. MalthusJAX provides an **abstract engine interface** (`AbstractEngine`) and a high-performance implementation (`GeneticEngine`) optimized for JAX/XLA and modern accelerators.

Key principles:

- **Separation of Concerns**: Genome/fitness (Level 1) and operators (Level 2) are composed into engines (Level 3) without modification.
- **Resource Budgeting**: RNG (PRNG) requirements are computed once at initialization via `ResourceMap`, enabling static allocation and precise "cascade" data flow.
- **Init-Phase Compilation**: Operators are "baked" with static input sizes at `init_state()`, allowing XLA to compile efficient kernels once and reuse them.
- **Immutability & Tracing**: All engine state is immutable (`@struct.dataclass`) and traced by JAX for reproducibility and safe parallelization.

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

1. **Compute ResourceMap** (`compute_resource_map`):
   - Calculates exact PRNG budget needed: selection + crossover + mutation + next-key.
   - Infers population flow: `Selection(N) → Parents(P) → Crossover(P/2) → Offspring(O) → Mutation(O) → Mutants(M)`.
   - Validates that operators are properly configured (input/output shapes).

2. **Bake Operators** (`OperatorState`):
   - Freezes operator input sizes in-place (e.g., `selection.set_input_length(pop_size)`).
   - This ensures XLA compiles the same kernel for every generation, avoiding recompilation overhead.

3. **Initialize Population**:
   - Call `PopulationClass.init_random(key, config, pop_size)` to create random genomes.
   - Evaluate initial population using the fitness evaluator.

4. **Enforce GSPMD Sharding** (optional):
   - Use `ShardingManager` to place population arrays on the correct devices/memory hierarchy.
   - For single-device setups, this optimizes layout; for multi-device, it ensures proper parallelism.

5. **Return GeneticEvolutionState**:
   - Bundles population, best genome, generation counter, RNG state, resource map, and baked operators.
   - This state is the **complete execution plan** for all subsequent steps.

**Key advantage**: All static setup happens once, enabling maximal XLA optimization and avoiding recompilation loops.

---

### Evolution Step: `step(state)`

Each generation executes in six phases (traced as named calls for HLO profiling):

```
Phase_0_Allocate_Entropy
  ↓
Phase_0a_Get_Active_Operators  (apply scheduled mutation strength, if any)
  ↓
Phase_1_Selection_Read          (select parents + extract elites)
  ↓
Phase_2_Reproduction_Fused      (crossover + mutation in one phase)
  ↓
Phase_3a_Merge                  (combine elites + mutants → next population)
  ↓
Phase_3b_Evaluate               (fitness evaluation)
  ↓
Phase_3c_Update_HOF             (update best genome / stagnation counter)
```

#### Phase_0_Allocate_Entropy
```python
all_keys = rmap.get_keys(state.rng_key)  # Split or fold master key
k_sel_slice   = all_keys[rmap.get_key_slice('selection')]
k_cross       = all_keys[rmap.get_key_slice('crossover')]
k_mut         = all_keys[rmap.get_key_slice('mutation')]
k_next        = all_keys[rmap.get_key_slice('next_key')][0]
```

**RNG Derivation Strategies**:
- **SPLIT** (default): `jax.random.split(key, n)` → Independent key streams (ideal for multi-device).
- **FOLD**: `jax.random.fold_in(key, index)` → Deterministic sequences (ideal for reproducibility).

User choice via `GeneticEngineParams.key_derivation: KeyDerivationStrategy`.

#### Phase_1_Selection_Read
```python
_, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
elites_genes = population[elite_idx].genes

selected_idx = operators.selection(k_sel_slice, population)  # returns indices
```

- Elite preservation: keeps the top `elitism` individuals.
- Selection: applies tournament, roulette, or other selection strategy.

#### Phase_2_Reproduction_Fused
```python
p1_pop = population[parent_indices[:num_pairs]]
p2_pop = population[parent_indices[num_pairs:]]

offspring_pop = operators.crossover(k_cross, p1_pop, p2_pop, config)
final_pop = operators.mutation(k_mut, offspring_pop, config)
```

- **Crossover**: Combines pairs of parents → `num_pairs * num_offspring` individuals.
- **Mutation**: Mutates offspring → `num_pairs * num_offspring * num_offspring_per_mutation` final mutants.
- **ResourceMap ensures shapes match**: Debug assertions validate that operators produce expected output counts.

#### Phase_3a_Merge
```python
next_genes = concatenate([elites_genes, mutants_genes[:remaining_slots]])
```

- Combines elites and truncated mutants to match original population size.

#### Phase_3b_Evaluate
```python
evaluated_pop = evaluator.evaluate_population(new_population)
```

- Batch fitness evaluation using `jax.vmap` internally.

#### Phase_3c_Update_HOF
```python
best_idx = jnp.argmax(evaluated_pop.fitness)
curr_best_fit = evaluated_pop.fitness[best_idx]
is_new = curr_best_fit > old_state.best_fitness

# Update stagnation counter (for termination logic)
next_state = state.replace(
    population=evaluated_pop,
    best_fitness=jnp.where(is_new, curr_best_fit, old_state.best_fitness),
    stagnation_counter=jnp.where(is_new, 0, old_state.stagnation_counter + 1),
    generation=old_state.generation + 1,
    rng_key=k_next
)
```

---

## 4) Resource Mapping & Cascade Data Flow

### The ResourceMap Contract

`ResourceMap` is a metadata-only (non-JAX-traced) structure that precomputes:

1. **Total RNG Budget**: Sum of keys needed across all operators.
2. **Per-Operator Allocations**: Start/end indices for key slices.
3. **Data Flow**: Input/output counts at each stage.

**Example** (pop_size=10):

```
Selection:     input=10  →  output=10 (parents)         [1 key]
Crossover:     input=10  →  output=10 (offspring)       [2 keys]
Mutation:      input=10  →  output=10 (mutants)         [2 keys]
Next-key:      output=1                                 [1 key]
────────────────────────────────────────────────────────
Total Budget: 6 keys
```

### Key Derivation Strategies

`KeyDerivationStrategy` enum allows users to choose:

| Strategy | Method | Use Case |
|----------|--------|----------|
| **SPLIT** | `jax.random.split(key, n)` | Multi-device, independent streams, less correlated noise |
| **FOLD** | `jax.random.fold_in(key, index)` | Reproducibility emphasis, deterministic sequences |

**Usage**:

```python
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    elitism=5,
    key_derivation=KeyDerivationStrategy.FOLD  # Choose strategy
)
```

**Implementation** (`ResourceMap.get_keys()`):

```python
def get_keys(self, key: chex.PRNGKey) -> chex.Array:
    """Derives keys using the configured strategy."""
    if self.key_derivation == KeyDerivationStrategy.SPLIT:
        return self.split_key(key)
    elif self.key_derivation == KeyDerivationStrategy.FOLD:
        return self.fold_key(key)
    else:
        raise ValueError(f"Unknown key derivation strategy: {self.key_derivation}")
```

---

## 5) Operator Baking & Static Input Lengths

### Why Baking Matters

Operators have a **static input length** (e.g., population size, number of pairs) that is known at initialization. By calling `.set_input_length(n)` once during `init_state()`, we:

1. **Enable operator-specific optimizations**: Selection can pre-allocate tournament arenas; crossover can pre-size buffers.
2. **Ensure XLA stability**: The kernel is compiled once and reused, avoiding recompilation per generation.
3. **Simplify the step loop**: No need to pass input length to every operator call.

**Example**:

```python
# In init_state()
active_sel = self.selection \
    .replace(num_selections=rmap.selection.output_count) \
    .set_input_length(rmap.selection.input_count)
    # ^ frozen at initialization; reused every generation

# In step()
selected_idx = active_sel(k_sel_slice, population)
# ^ uses pre-baked input_length, no recompilation
```

---

## 6) Scheduled Mutation Strength

Engines can optionally apply a **time-dependent mutation schedule** to adapt exploration over generations.

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

**Implementation** (in `_get_active_operators()`):

```python
if self.engine_params.mutation_strength_schedule is not None:
    scheduled_strength = self.engine_params.mutation_strength_schedule(state.generation)
    updated_mutation = operators.mutation.replace(mutation_strength=scheduled_strength)
    return operators.replace(mutation=updated_mutation)
```

---

## 7) Ask/Tell Interface

For decoupled evaluation workflows (e.g., external simulators, distributed evaluation), the engine provides an **ask/tell** pattern:

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
- `tell()` completes the evolutionary step using the returned, pre-evaluated population.

---

## 8) Sharding & Multi-Device Layout

`ShardingManager` enforces GSPMD sharding layouts for optimal data placement:

```python
sharding_mgr = ShardingManager(axis_name='batch')

# Population matrices (batch-major)
matrix_sharding = NamedSharding(mesh, P('batch', None))

# Fitness vectors
vector_sharding = NamedSharding(mesh, P('batch'))

# Replicated scalars (best_fitness, etc.)
replicated_sharding = NamedSharding(mesh, P())
```

**For single-device**: Optimizes memory layout and L-cache alignment.
**For multi-device**: Ensures proper data distribution across shards.

---

## 9) Extension Points for Custom Engines

Developers can extend `AbstractEngine` to implement:

1. **Custom selection logic**: Override `_selection_phase()`.
2. **Alternative reproduction strategies**: Replace `_reproduction_phase()` with, e.g., asexual reproduction or island models.
3. **Custom metrics**: Extend `AbstractGenerationOutput` to track domain-specific KPIs.
4. **Custom state management**: Subclass `AbstractEvolutionState` to track additional metadata (e.g., diversity, lineage).

**Example**:

```python
@struct.dataclass
class IslandEvolutionState(AbstractEvolutionState[...]):
    """State for island model."""
    islands: Tuple[BasePopulation[...], ...] = struct.field()
    migration_history: chex.Array = struct.field()

class IslandEngine(AbstractEngine[...]):
    def step(self, state: IslandEvolutionState) -> Tuple[IslandEvolutionState, AbstractGenerationOutput]:
        # Implement island-specific logic
        ...
```

---

## 10) Best Practices & Performance Tips

### General

- **Use immutable state**: Always rely on `state.replace(...)` to update engine state.
- **Avoid Python control flow in JIT**: Keep loops and conditions at the Python level, outside the `step()` method.
- **Profile with JAX debugging**: Use `jax.named_call` markers (already in `GeneticEngine`) and JAX's profiler for HLO inspection.

### Resource Management

- **Compute ResourceMap early**: Call `compute_resource_map()` once in `init_state()` and store in `GeneticEvolutionState`.
- **Monitor RNG budget**: Ensure operators respect their allocated key slices; debug assertions validate this.
- **Choose key derivation wisely**:
  - Use `SPLIT` for multi-device to ensure independent streams per device.
  - Use `FOLD` when reproducibility and seeding are paramount.

### Sharding

- **For single-device**: Let `ShardingManager` optimize memory layout; the overhead is negligible.
- **For multi-device**: Ensure the batch axis (`axis_name='batch'`) aligns with your device mesh dimensions.
- **Replicate scalars**: Non-batched values (best_fitness, generation) should use `replicated_sharding`.

### Mutation Scheduling

- **Define schedules as pure functions**: `schedule(generation: int) -> float`.
- **Test schedule output**: Ensure the returned values are valid and in expected ranges.
- **Combine with elitism**: Scheduled mutation works best with elitism to preserve good solutions.

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
| `GeneticEvolutionState` | Complete state bundle (population, HOF, RNG, ResourceMap) | `@struct.dataclass` | ✅ |
| `ResourceMap` | Precomputed RNG budget & data flow | `@struct.dataclass` (no pytree) | ✅ |
| `KeyDerivationStrategy` | SPLIT vs FOLD RNG derivation | Enum | N/A |
| `ShardingManager` | GSPMD sharding layout manager | Python class | ✅ (stateless) |
| `OperatorState` | Baked operators with frozen input sizes | `@struct.dataclass` | ✅ |

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
