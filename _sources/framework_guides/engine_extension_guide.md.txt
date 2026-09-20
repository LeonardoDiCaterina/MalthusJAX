# Engines in MalthusJAX – Outline

## 📚 Overview
- The **Engine** (`src/malthusjax/engine/`) is the ultimate orchestrator in MalthusJAX. 
- It ties together the Genomes, Populations, Fitness Evaluators, Operators, and Resource Mapper into a single, fully JIT-compiled loop via `jax.lax.scan`.
- If you need a completely custom algorithmic flow (e.g., an Island Model, Co-evolution, or highly bespoke Reinforcement Learning loops), you subclass `AbstractEngine`.

---

## 1️⃣ Core Abstractions

All engines rely on three primary abstractions found in `engine/base.py`:

1. **`AbstractEngineParams`**: An immutable dataclass holding static configuration (`pop_size`, `num_generations`, `elitism`). JAX treats this as completely static (`pytree_node=False`), so any changes trigger recompilation.
2. **`AbstractEvolutionState`**: The mutable PyTree passed *through* the `lax.scan` loop. It holds the `population`, `generation` counter, `best_fitness`, and the master `rng_key`.
3. **`AbstractGenerationOutput`**: The KPI payload returned at every step. XLA automatically stacks these across generations, giving you a full historical dashboard `(num_generations, ...)` at the end of the run.

---

## 2️⃣ Phase 1: `init_state` (Compilation & Allocation)
Before evolution begins, the engine runs `init_state(self, rng_key)`.
This is a critical, one-time setup phase where:
- The **Resource Mapper** is invoked to pre-calculate the massive, static key arrays for all operators.
- The initial `Population` is instantiated.
- The initial population is evaluated to establish baseline fitness scores.
- The resulting `EvolutionState` is cached so `run()` can be executed repeatedly with zero compilation overhead.

---

## 3️⃣ Phase 2: `step` (The 5-Phase Generation)
The `step(self, state)` method defines the actual logic for a single generation.
In the flagship `GeneticEngine`, this strictly follows 5 traceable phases:

1. **Entropy Allocation**: The Resource Mapper slices the master key into specific buffers.
2. **Selection**: Parents are chosen from the population (via tournament, roulette, etc.).
3. **Reproduction (Crossover & Mutation)**: Offspring are generated using the pre-allocated PRNG keys.
4. **Merge**: The new offspring (and any preserved elites) are merged to form the new population matrix.
5. **Evaluation**: The new population is evaluated via `vmap`.

*Note: For debugging XLA HLO profiles, MalthusJAX allows you to enable `jax.named_call` tracing on these 5 phases, though it is disabled by default to allow XLA to fuse them into a single blazing-fast kernel.*

---

## 4️⃣ Phase 3: `run` (The Scan Loop)
You rarely need to override `run()`. The `AbstractEngine` implements it using `jax.lax.scan` over the `step` method. This compiles the entire evolution loop—thousands of generations—into a single GPU/TPU execution call.

> [!NOTE]
> For a comprehensive deep dive into how these methods are implemented in the flagship engine—including advanced asynchronous execution via the `ask()` and `tell()` loop—see the **[GeneticFastEngine Guide](file:///Users/leonardodicaterina/.gemini/antigravity-ide/brain/ca431008-9110-4905-9509-fbd0ec89a09b/genetic_fastengine_guide.md)**.

---

## 5️⃣ Extending to Specialized Models (QD, MO, Island)
The `AbstractEngine` is designed to be subclassed for entirely different evolutionary paradigms. MalthusJAX provides native implementations for the three most common specialized architectures:

### 1. Quality-Diversity (`src/malthusjax/engine/qd/`)
The `MapElitesEngine` drops standard mutation/crossover logic in favor of a specialized Quality-Diversity (QD) approach.
- **How it works**: It relies strictly on a `BaseEmitter` (like `TensorNEATEmitter`) and a `BaseQDEvaluator`.
- **The State (`MapElitesState`)**: Instead of a flat population matrix, it carries a high-dimensional `repertoire` (a multi-dimensional grid of elite solutions) across the `lax.scan` loop. It leverages `qdax` internally for this grid storage.

**Example: Running MAP-Elites**
```python
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams

# Notice how we pass an Emitter instead of crossover/mutation operators
qd_engine = MapElitesEngine(
    emitter=my_tensorneat_emitter,
    evaluator=my_qd_evaluator,
    engine_params=MapElitesEngineParams(pop_size=128, num_generations=1000)
)
```

### 2. Multi-Objective (`src/malthusjax/engine/mo/`)
The `MOEngine` implements the classic NSGA-II paradigm by intercepting the standard `step` function right after the "Merge" phase.
- **How it works**: It requires a `BaseMOEvaluator` (which returns a vector of fitnesses instead of a scalar). 
- **The Upgrade**: The engine wraps the standard population in an `MOPopulation`. After evaluating new offspring and merging them with the elites, it automatically triggers **Non-Dominated Sorting** and **Crowding Distance** calculations. This perfectly truncates the mixed $\mu+\lambda$ population back down to `pop_size` while preserving Pareto fronts!

**Example: Running NSGA-II**
```python
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams

mo_engine = MOEngine(
    emitter=my_standard_emitter, # Orchestrates mutation/crossover
    evaluator=my_mo_evaluator,   # Returns [obj_1, obj_2]
    engine_params=MOEngineParams(pop_size=100, num_generations=500)
)
```

### 3. Island Models (`src/malthusjax/engine/island_model/`)
Island models run multiple isolated populations in parallel, periodically migrating individuals between them to maintain genetic diversity and prevent premature convergence.
- **The Meta-Engine**: `BaseIslandModel` acts as a wrapper that seamlessly upgrades any standard 1D `BaseEngine` into a 2D Island Model by wrapping the engine's `init` and `step` methods in a `jax.vmap` across the `num_islands` axis.
- **Migration Topologies**: Subclasses like `RingTopologyIsland` implement the `migrate()` method, handling the complex JAX indexing required to shuffle elite individuals across island boundaries without breaking XLA tracing.

**Example: Ring Topology Island Model**
```python
from malthusjax.engine.island_model.topologies import RingTopologyIsland

# Wrap a standard 1D GeneticEngine to run across 4 isolated islands!
island_engine = RingTopologyIsland(
    engine=my_standard_genetic_engine,
    num_islands=4,             # Run 4 engines in parallel via vmap
    migration_interval=50,     # Swap genomes every 50 generations
    num_migrants=5             # Move 5 individuals per interval
)
```

---

## 6️⃣ Registering a Custom Engine
If you write a completely new Engine class, you can register it for the Composer using the `@register_engine` decorator.

```python
from malthusjax.composer.decorators import register_engine
from malthusjax.engine.base import AbstractEngine

@register_engine("my_custom_engine")
def build_custom_engine(**kwargs) -> AbstractEngine:
    # Parse kwargs into your custom EngineParams
    # Return instantiated Engine
    ...
```

Once registered, you can summon it from your TOML files:
```toml
[pipelines.my_pipeline]
engine_type = "my_custom_engine"
```
