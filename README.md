# MalthusJAX

[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](https://github.com/LeonardoDiCaterina/MalthusJAX)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Evolve solutions at GPU speed.** MalthusJAX is a JAX-powered evolutionary computation framework -- define your experiment in a TOML file, and run multi-seed, hardware-accelerated benchmarks with a single function call.

📖 **[Read the full documentation](https://malthusjax.readthedocs.io/)** for architecture details, API reference, and tutorials.

No boilerplate. No recompilation between generations. Just fast evolution.

---

### Data-Driven Evaluators (v3.0+)

MalthusJAX now safely handles external data files directly interacting with JIT-compiled optimization runs.

```toml
[experiment.shared]
fitness = "tsp:data_id=benchmark_52"

[data.benchmark_52]
source = "file"
path = "data/tsp/berlin52.tsp"  # TSPLIB benchmark instance
```
```python
result = load_experiment_config("config.toml")
composer = Composer()
exec_result = composer.quick_run(data_config=result.data_registry, **result.pipelines['ga_baseline'])
```
See the [Option C Guide](OPTION_C_GUIDE.md) for more details.

## Installation

```sh
# Install directly from GitHub
pip install git+https://github.com/LeonardoDiCaterina/MalthusJAX.git

# Or clone & install in development mode
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX
make install-dev
```

Python 3.8+ required.

---

## The Fastest Way: TOML + 3 Lines of Python

Describe your experiment in a simple config file:

```toml
# experiment.toml
[experiment]
name       = "sphere_test"
output_dir = "results/sphere"

[experiment.shared]
fitness       = "sphere:dim=10"
genome_type   = "real"
genome_length = 10
bounds        = [-5.0, 5.0]
pop_size      = 64
generations   = 100
seeds         = [42, 43, 44]

[pipelines.ga]
engine_type = "ga"
selection   = "tournament:num_selections=64,tournament_size=3"
crossover   = "simulated_binary:eta=15"
mutation    = "gaussian:mutation_rate=0.1"
```

Then run it:

```python
from malthusjax.composer import Composer

result = Composer.from_toml("experiment.toml")
print(result.summary_table())
result.plot_convergence()
```

That's it -- multi-seed execution, result aggregation, and convergence plots handled automatically.

---

## Quick Experiment in Pure Python
Don't want a config file? Use `quick_run()` to launch an experiment in one call:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=64,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
)

print(result.aggregated_summary())
```

Operators are specified as readable strings like `"gaussian:mutation_rate=0.1"` -- no need to import individual classes.

---

## Compare Algorithms Side-by-Side

Want to know which crossover strategy works best? Compare them with aligned seeds for a fair test:

```python
cmp = composer.compare(
    pipelines={
        "Blend + Gaussian": dict(
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1",
        ),
        "SBX + Polynomial": dict(
            crossover="simulated_binary:eta=2",
            mutation="polynomial:mutation_rate=0.1",
        ),
        "Evosax SimpleGA": dict(
            backend="evosax",
            evosax_strategy="SimpleGA",
        ),
    },
    fitness="sphere:dim=10",
    pop_size=50,
    generations=100,
    seeds=(42, 43),
)

cmp.summary_table()      # Aggregated metrics per pipeline
cmp.plot_convergence()   # Overlay convergence curves
```

Cross-framework comparison with [evosax](https://github.com/RobertTLange/evosax) is built in -- just set `backend="evosax"`.

---

## Beyond Continuous Optimization: Flexible Solution Representations

Unlike evosax (limited to real-valued vectors), MalthusJAX lets you choose the genome type that matches your problem:

- **Real-valued genomes** (`genome_type="real"`) — Continuous optimization (Sphere, Rastrigin, BBOB)
- **Binary genomes** (`genome_type="binary"`) — Combinatorial problems (knapsack, satisfiability)
- **Categorical genomes** (`genome_type="categorical"`) — Permutation problems (TSP, scheduling)

Each genome type gets **problem-specific operators**:
- Real: Gaussian/polynomial mutation, blend/SBX crossover
- Binary: Bit-flip mutation, uniform/single-point crossover
- Categorical: Swap/scramble mutation, order-preserving crossover

### Example: 0/1 Knapsack (Binary Genome)

Solve a combinatorial optimization problem with a **binary genome**. The Knapsack evaluator uses the same **unified data interface as TSP** — either generate random instances with `create_synthetic()` or load pre-computed weights/values:

**Option 1: Generate random knapsack instances (synthetic data)**
```python
import jax.random as jar
from malthusjax.core.fitness.binary_evaluators import KnapsackEvaluator
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.operators.selection import ElitePoolSelection
from malthusjax.operators.crossover import UniformCrossover
from malthusjax.operators.mutation import BitFlipMutation
from malthusjax.engine import GeneticEngine, GeneticEngineParams

# Create a random knapsack problem with synthetic data
evaluator = KnapsackEvaluator.create_synthetic(
    n_items=20,
    capacity_ratio=0.5,
    seed=42,
    maximize=True  # Maximize total value
)

# Set up the engine
genome_config = BinaryGenomeConfig(length=20)  # 20-bit binary string
engine = GeneticEngine(
    engine_params=GeneticEngineParams(pop_size=64, elitism=2, num_generations=100),
    genome_config=genome_config,
    evaluator=evaluator,
    selection=ElitePoolSelection(num_selections=32, elite_k=2),
    crossover=UniformCrossover(num_offspring=2),
    mutation=BitFlipMutation(mutation_rate=0.05),
)

# Run optimization
key = jar.PRNGKey(42)
state = engine.init_state(rng_key=key)
final_state, history, elapsed = engine.run(state, time_it=True)

print(f"Best value: {final_state.best_fitness:.2f}")
```

**Option 2: Load pre-computed weights and values**
```python
import jax.numpy as jnp
from malthusjax.core.fitness.binary_evaluators import KnapsackEvaluator

# Pre-computed item data
weights = jnp.array([5.0, 3.0, 7.0, 2.0, 4.0])
values = jnp.array([12.0, 8.0, 20.0, 5.0, 10.0])
capacity = 12.0  # Max weight

# Load evaluator with your data
evaluator = KnapsackEvaluator.create_from_data(
    weights=weights,
    values=values,
    capacity=capacity,
    maximize=True
)

# Then use with GeneticEngine (same as above)
```

**Unified Data Interface:**
Both TSP and Knapsack now follow the same pattern:
- `create_synthetic()` — Generate random instances procedurally
- `create_from_data()` — Load pre-computed problem data

| Problem | Synthetic Method | Data Container |
|---------|------------------|-----------------|
| **TSP** | `TSPEvaluator.create_synthetic(num_cities=50, seed=42)` | distance_matrix |
| **Knapsack** | `KnapsackEvaluator.create_synthetic(n_items=50, seed=42)` | KnapsackData(weights, values) |

**Genome Types in MalthusJAX:**

| Type | Encoding | Used For | Examples |
|------|----------|----------|----------|
| `real` | Continuous values in [bounds] | Continuous optimization, TSP (Random Key) | Sphere, Rastrigin, TSP, BBOB |
| `binary` | Bit strings (0/1 decisions) | Combinatorial selection | Knapsack, feature selection, SAT |
| `categorical` | Permutations/discrete choices | Ordering problems | (Planned: scheduling, task assignment) |

Each genome type gets **problem-specific operators**:
- **Real**: Gaussian/polynomial mutation, blend/SBX crossover
- **Binary**: Bit-flip mutation, uniform/single-point crossover
- **Categorical**: Swap/scramble mutation (planned)

---

## Data-Driven Optimization: Load External Benchmarks

MalthusJAX seamlessly integrates external data through the `data_config` parameter — pass problem-specific data directly into JIT-compiled optimization runs. Classic example: **Traveling Salesman Problem (TSP)**:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

result = composer.quick_run(
    fitness="tsp:data_id=berlin52",
    data_config={
        "berlin52": {
            "source": "synthetic",
            "num_cities": 50,
            "random_seed": 42,
        },
    },
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    pop_size=50,
    generations=100,
    seeds=(42, 43, 44),
    genome_type="real",  # TSP uses Real-valued encoding (Random Key)
)

summary = result.aggregated_summary()
print(f"Best tour: {summary['best_fitness']['mean']:.2f} units")
```

For TOML-based reproducible experiments with external data:

```toml
[experiment.shared]
fitness = "tsp:data_id=berlin52"
genome_type = "real"
pop_size = 64
generations = 200
seeds = [42, 43, 44]

[data.berlin52]
source = "file"
path = "data/tsp/berlin52.tsp"

[pipelines.ga]
selection = "tournament:num_selections=25"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"
```

Data sources supported:
- **Files** — Load pre-computed TSP instances (`.tsp` files from [TSPLIB](http://www.math.uwaterloo.ca/tsp/)), distance matrices, or other benchmarks
- **Synthetic** — Procedurally generate random problem instances (TSP with random coordinates, BBOB functions, etc.)

**Note on `.tsp` files:** These are standard problem files from the TSPLIB benchmark suite. They contain city coordinates or distance matrices in plain text. Classic instances include `berlin52`, `bier127`, `ch130`, and many others. Free instances are available at http://www.math.uwaterloo.ca/tsp/

---

## Analysis & Results: From Runs to Insights

Every experiment automatically aggregates results across seeds and computes statistics:

```python
# Multi-seed aggregation
summary = result.aggregated_summary()
print(f"Best:  {summary['best_fitness']['mean']:.4f} ± {summary['best_fitness']['stdev']:.4f}")
print(f"Final: {summary['mean_fitness']['mean']:.4f} ± {summary['mean_fitness']['stdev']:.4f}")

# Export convergence history to pandas for analysis
import pandas as pd
history = result.combined_history()
df = pd.DataFrame(history)

# Analyze per-seed behavior
by_seed = df.groupby('seed')['best_fitness'].describe()
print(by_seed)

# Plot convergence across all runs
import matplotlib.pyplot as plt
for seed in df['seed'].unique():
    seed_data = df[df['seed'] == seed]
    plt.plot(seed_data['generation'], seed_data['best_fitness'], label=f"Seed {seed}", alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.legend()
plt.show()
```

For multi-algorithm comparison:

```python
# Overlay convergence curves for fair comparison
comparison = composer.compare(
    pipelines={
        "GA+Blend": {...},
        "GA+SBX": {...},
    },
    fitness="sphere:dim=10",
    seeds=(42, 43, 44),
)

# Side-by-side metrics
print(comparison.summary_table())

# Plot overlaid convergence (one per seed)
for i in range(3):
    comparison.plot_convergence(seed_index=i)
```

---

## Full Control: Build Your Own Engine

For researchers who want to control every component:

```python
import jax.random as jar
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.operators.selection import ElitePoolSelection
from malthusjax.operators.crossover import SimulatedBinaryCrossover
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.engine import GeneticEngine, GeneticEngineParams

# Define the problem
genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
evaluator = BBOBEvaluator.create(
    BBOBConfig(fn_name="sphere", num_dims=10, maximize=False)
)

# Pick your operators
selection = ElitePoolSelection(num_selections=20, elite_k=2)
crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
mutation  = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

# Assemble and run
engine = GeneticEngine(
    engine_params=GeneticEngineParams(pop_size=32, elitism=2, num_generations=100),
    genome_config=genome_config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
)

key = jar.PRNGKey(42)
state = engine.init_state(rng_key=key)
final_state, history, elapsed_time = engine.run(state, time_it=True)

print(f"Best fitness: {final_state.best_fitness:.6f}")
print(f"Elapsed time: {elapsed_time:.6f} seconds")
```

Every operator is a JIT-compilable callable -- swap any component and the engine recompiles once, then runs at full speed.

---

## Ask/Tell: Plug In External Evaluators

Need to evaluate fitness outside JAX (physics simulators, cloud services, human-in-the-loop)?

```python
state = engine.init_state(key)

for i in range(100):
    # ASK for the current population
    engine_with_entropy, population = engine.ask(state)

    # Evaluate however you want
    evaluated_pop = my_external_simulator(population)

    # TELL the engine the results
    state = engine_with_entropy.tell(state, evaluated_pop)

print(f"Final: {state.best_fitness:.6f}")
```

---

## Available Operators

Mix and match from the built-in catalog:

| Category | Operators |
|----------|-----------|
| **Selection** | `tournament`, `roulette`, `elite_pool` |
| **Real Crossover** | `uniform_real`, `blend` (BLX-alpha), `simulated_binary` (SBX), `binomial` |
| **Binary Crossover** | `uniform_binary`, `single_point` |
| **Real Mutation** | `gaussian`, `ball`, `polynomial` |
| **Binary Mutation** | `bitflip`, `scramble`, `swap` |
| **Fitness Functions** | `sphere`, `rastrigin`, `griewank`, `bbob` (24 BBOB functions), `knapsack`, `binary_sum` |

All operators support string specs: `"tournament:num_selections=50,tournament_size=3"`.

---

## How It Works

MalthusJAX is built in three layers you can use independently:

1. **Core** -- Genomes (real, binary, categorical) and fitness evaluators. These are the building blocks.
2. **Operators** -- Selection, crossover, and mutation. Each is a pure function that JIT-compiles cleanly.
3. **Engine** -- Wires everything together into an evolution loop. Compiles once, then runs every generation as a single fused GPU kernel.

The **Composer** sits on top and lets you skip the wiring -- describe what you want in strings or TOML and it builds the engine for you.

---

## Extending MalthusJAX

Creating a custom operator is straightforward -- implement the core logic and batching + JIT compilation are inherited automatically:

```python
import jax
from flax import struct
from malthusjax.operators.mutation import BaseMutation

@struct.dataclass
class MyMutation(BaseMutation):
    num_offspring: int = struct.field(pytree_node=False)
    strength: float    = struct.field(pytree_node=False)

    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _generate_noise(self, keys, cfg):
        return jax.random.normal(keys[0], shape=cfg.shape) * self.strength

    def _mutate_one(self, genome, noise, cfg):
        return genome.replace(values=genome.values + noise)
```

Drop it into any pipeline -- the engine handles the rest.

---

## Development

```bash
make check-all    # Lint + format + type-check + test (80% coverage minimum)
make test         # Run tests only
# Ensure 80% test coverage is maintained across the codebase
make lint         # Ruff linting
make docs         # Build Sphinx documentation
```

---

---

## License

[MIT License](LICENSE)