# GeneticEngine Instantiation Guide

## Complete __init__ Signature

The `GeneticEngine` is a **flax.struct.dataclass** with the following constructor signature:

```python
GeneticEngine(
    # Required parameters (genome/fitness)
    genome_config: Any,
    evaluator: BaseEvaluator,
    
    # Required parameters (operators)
    selection: BaseSelection,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    
    # Required parameter (engine configuration)
    engine_params: GeneticEngineParams,
    
    # Optional parameters
    enable_progress_bar: bool = False,
)
```

**Key Point**: All parameters are **keyword-only** (no positional arguments). The class is immutable.

---

## Complete Working Example

Here's a fully working instantiation pattern from the test suite:

```python
import jax.random as jar
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.operators.selection import ElitePoolSelection
from malthusjax.operators.crossover import SimulatedBinaryCrossover
from malthusjax.operators.mutation import GaussianMutation

# 1. Configure genome
genome_config = RealGenomeConfig(
    shape=(10,),           # 10-dimensional problem
    bounds=(-5.0, 5.0)     # Search space [-5, 5]
)

# 2. Configure fitness evaluator
bbob_config = BBOBConfig(
    fn_name="sphere",      # Problem: minimize sphere function
    num_dims=10,
    maximize=False         # We want to minimize
)
evaluator = BBOBEvaluator.create(bbob_config)

# 3. Configure operators
selection = ElitePoolSelection(
    num_selections=100,    # Must match population size
    elite_k=3              # Preserve 3 best individuals
)

crossover = SimulatedBinaryCrossover(
    num_offspring=2,       # Create 2 offspring per crossover event
    eta=15.0               # Distribution index
)

mutation = GaussianMutation(
    num_offspring=1,       # Mutate each individual
    mutation_rate=0.1,     # 10% probability per gene
    mutation_strength=0.5  # Perturbation standard deviation
)

# 4. Configure engine parameters
engine_params = GeneticEngineParams(
    pop_size=100,                    # Population size
    elitism=2,                       # Keep 2 best unchanged
    num_generations=50,              # Run for 50 generations
    # Optional advanced parameters:
    # key_derivation=KeyDerivationStrategy.SPLIT,     # default
    # prng_impl=PRNGImpl.THREEFRY,                     # default
    # schedule_type=ScheduleType.CONSTANT,            # default
    # track_best=TrackBest.LIGHT,                     # default
    # initial_strength=0.1,                           # for schedules
    # final_strength=0.0,                             # for schedules
    # debug_tracing=False,                            # default
)

# 5. Create engine
engine = GeneticEngine(
    engine_params=engine_params,       # ← Passed as named parameter
    genome_config=genome_config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    enable_progress_bar=True,          # Optional: show progress bar
)

# 6. Initialize and run
key = jar.PRNGKey(42)
state = engine.init_state(key)
final_state, history = engine.run(state)

print(f"Best fitness: {final_state.best_fitness}")
```

---

## GeneticEngineParams Structure

`GeneticEngineParams` is a dataclass that inherits from `AbstractEngineParams`:

### Inherited Parameters (from AbstractEngineParams)
```python
pop_size: int
    Population size. Must be > 0.
    GPU efficiency: Powers of 2 preferred (32, 64, 128, 256).
    Validation: pop_size > 0 (checked in validate_engine_params)
    Example: pop_size=100

elitism: int
    Number of elite individuals preserved each generation.
    Valid range: 0 ≤ elitism < pop_size
    - 0: No elitism (all individuals bred)
    - 1-5: Typical (preserve 1-5% of population)
    Example: elitism=2

num_generations: int
    Number of evolutionary cycles to run.
    Baked into JIT-compiled code; changing triggers recompilation.
    Must be > 0.
    Example: num_generations=100

unroll_num: int
    JAX scan unroll factor (default: 1).
    Historically used for latency/memory trade-off.
    Always defaults to 1 now (deprecated).
```

### GeneticEngineParams-Specific Parameters

```python
key_derivation: KeyDerivationStrategy
    RNG splitting strategy for entropy allocation.
    - SPLIT: Traditional jax.random.split (uncorrelated, default)
    - FOLD: jax.random.fold_in (deterministic, vectorizable)
    Default: KeyDerivationStrategy.SPLIT
    Example: key_derivation=KeyDerivationStrategy.SPLIT

prng_impl: PRNGImpl
    PRNG type for converting integer seeds to typed keys.
    - THREEFRY: Deterministic on all platforms (default)
    - PHILOX: Fast on GPU, may vary cross-platform
    Default: PRNGImpl.THREEFRY
    Example: prng_impl=PRNGImpl.THREEFRY

schedule_type: ScheduleType
    Mutation strength schedule across generations.
    - CONSTANT: Fixed strength (no schedule, default)
    - LINEAR_DECAY: Linearly fade from initial_strength → final_strength
    - COSINE_ANNEAL: Cosine annealing (smooth decay)
    - EXPONENTIAL_DECAY: Exponential decay
    Default: ScheduleType.CONSTANT
    Example: schedule_type=ScheduleType.LINEAR_DECAY

track_best: TrackBest
    Hall-of-Fame tracking strategy for best individual.
    - NONE: No tracking; compute best_genome once at end (fastest)
    - LIGHT: Track best_fitness as running max, genome not tracked
    - FULL: Track both best_fitness and best_genome every generation (slowest)
    Default: TrackBest.LIGHT
    Example: track_best=TrackBest.LIGHT

initial_strength: float
    Mutation strength at generation 0 (used by non-CONSTANT schedules).
    Typical range: [0.05, 0.5]
    Default: 0.1
    Example: initial_strength=0.2

final_strength: float
    Mutation strength at generation = num_generations - 1.
    Typical range: [0.0, 0.1]
    Default: 0.0
    Example: final_strength=0.01

debug_tracing: bool
    Enable jax.named_call labels for XLA HLO profiling.
    - False (default): Allows XLA to fuse all 5 phases into one kernel
    - True: Adds phase-level labels visible in profilers
    Default: False
    Example: debug_tracing=True
```

---

## How engine_params Is Used

### 1. **Passed to GeneticEngine Constructor**

```python
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    elitism=2
)

engine = GeneticEngine(
    engine_params=engine_params,  # ← Keyword parameter!
    genome_config=...,
    evaluator=...,
    selection=...,
    crossover=...,
    mutation=...,
)
```

### 2. **Stored as Immutable Field**

`engine_params` is stored in the `GeneticEngine` instance as a non-pytree field:
- Immutable (cannot be modified without creating new engine)
- Static (changes trigger JIT recompilation)
- Used by `init_state()` to bake operators and resource map

### 3. **Used During Initialization**

In `init_state()`, the engine reads from `self.engine_params`:
```python
def init_state(self, rng_key: Union[int, chex.Array]) -> GeneticEvolutionState:
    params = cast(GeneticEngineParams, self.engine_params)   # Type cast
    
    # Use params.pop_size to initialize population
    population = self.genome_config.init_population(
        init_pop_key, 
        self.engine_params.pop_size  # ← Used here
    )
    
    # Use params.elitism for operator configuration
    active_sel = self.selection.set_n_elites(params.elitism)
    
    # Use other params for resource map computation
    ...
```

### 4. **Used During Evolution**

In `step()`, params are accessed for scheduling and evolution logic:
```python
def step(self, state: AbstractEvolutionState[...]):
    params = cast(GeneticEngineParams, self.engine_params)
    
    # Use elitism count for merge phase
    if params.elitism > 0:
        elites = ...
    
    # Use scheduling parameters for mutation strength
    strength = compute_scheduled_strength(
        generation=state.generation,
        num_generations=params.num_generations,
        initial_strength=params.initial_strength,
        final_strength=params.final_strength,
        schedule_type=params.schedule_type
    )
```

---

## Factory Method Alternative

Instead of creating `GeneticEngine` directly, you can use the factory method which handles more configuration automatically:

```python
from malthusjax.composer.engine_factory import build_engine

adapter = build_engine(
    fitness_evaluator=evaluator,
    selection_op=selection,
    crossover_op=crossover,
    mutation_op=mutation,
    # Configuration (defaults shown)
    genome_type="real",          # "real" or "binary"
    pop_size=50,
    generations=100,
    elitism=2,
    genome_shape=(10,),          # tuple of ints
    bounds=(-5.0, 5.0),
    unroll_factor=1,
    enable_progress_bar=False,
    # Optional: inject initial population for reproducibility
    initial_population=None,
    # Optional: specify PRNG implementation
    prng_impl="threefry",        # "threefry" or "philox"
    maximize=False,              # Inferred from evaluator if not provided
)

# This returns GeneticEngineAdapter (wraps GeneticEngine)
# Can run directly:
results = adapter.run_once(key)
```

**Note**: `build_engine` wraps the raw `GeneticEngine` in a `GeneticEngineAdapter`, which is needed for compatibility with the BenchmarkRunner protocol and proper output sign handling.

---

## Parameter Validation

Always validate engine parameters before running evolution:

```python
from malthusjax.engine.base import validate_engine_params

engine_params = GeneticEngineParams(
    pop_size=100,
    elitism=5,
    num_generations=50
)

# This will raise ValueError if params are invalid:
validate_engine_params(engine_params)
# Checks:
# - pop_size > 0
# - num_generations > 0
# - 0 <= elitism < pop_size

engine = GeneticEngine(
    engine_params=engine_params,
    ...
)
```

---

## Common Mistakes

### ❌ Mistake 1: Using positional arguments
```python
# WRONG
engine = GeneticEngine(
    genome_config,     # ← This fails!
    evaluator,
    selection,
    crossover,
    mutation,
    engine_params
)
```

### ✅ Correct: Use keyword arguments
```python
engine = GeneticEngine(
    genome_config=genome_config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    engine_params=engine_params,
)
```

---

### ❌ Mistake 2: Passing engine_params as `params`
```python
# WRONG (old API)
engine = GeneticEngine(
    genome_config=genome_config,
    params=engine_params,  # ← Wrong parameter name!
    ...
)
```

### ✅ Correct: Use `engine_params` (not `params`)
```python
engine = GeneticEngine(
    genome_config=genome_config,
    engine_params=engine_params,  # ← Correct!
    ...
)
```

---

### ❌ Mistake 3: Invalid elitism
```python
# WRONG: elitism must be < pop_size
engine_params = GeneticEngineParams(
    pop_size=50,
    elitism=50  # ← Fails! Must be < 50
)
```

### ✅ Correct: elitism < pop_size
```python
engine_params = GeneticEngineParams(
    pop_size=50,
    elitism=2  # ← OK: 0 <= 2 < 50
)
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Constructor** | `GeneticEngine(...)` with keyword-only parameters |
| **engine_params location** | Passed as `engine_params=GeneticEngineParams(...)` |
| **engine_params type** | `GeneticEngineParams` (dataclass extending `AbstractEngineParams`) |
| **engine_params storage** | Stored in `self.engine_params` field (immutable, non-pytree) |
| **engine_params modification** | Use `.replace()` to create modified copy; triggers recompilation |
| **Required for instantiation** | Yes, always required |
| **Factory alternative** | `build_engine()` from `malthusjax.composer.engine_factory` |
| **Validation** | Call `validate_engine_params()` before `init_state()` |
