# Quick Start

Welcome to MalthusJAX! This guide will get you up and running with your first evolutionary computation experiments in minutes.

## Installation

MalthusJAX requires Python 3.9+ and JAX.

To install the framework from PyPI or from the local source, run:

```bash
pip install git+https://github.com/LeonardoDiCaterina/MalthusJAX.git

# Or clone & install in development mode
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX
```

If you plan to contribute or run the test suite, use the development installation:

```bash
make install-dev
```

## Running Your First Experiment (TOML)

MalthusJAX's Composer allows you to define complex, multi-seed benchmarks using a single declarative TOML file.

Create a file named `experiment.toml`:

```toml
[experiment]
name       = "sphere_test"
output_dir = "results/sphere"

[experiment.shared]
fitness       = "sphere:dim=10"
genome_type   = "real"
genome_length = 10
bounds        = [ -5.0, 5.0 ]
pop_size      = 64
generations   = 100
seeds         = [ 42, 43, 44 ]

[pipelines.ga]
engine_type = "ga"
selection   = "tournament:num_selections=64,tournament_size=3"
crossover   = "simulated_binary:eta=15"
mutation    = "gaussian:mutation_rate=0.1"
```

Then, load and execute the experiment in Python:

```python
from malthusjax.composer import Composer

result = Composer.from_toml("experiment.toml")
print(result.summary_table())
result.plot_convergence()
```

The Composer automatically handles the multi-seed runner, artifact logging, and JIT compilation.

## Running via the Python API

If you prefer to configure experiments dynamically in Python without a TOML file, you can use the `quick_run` method:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

result = composer.quick_run(
    seeds=[42, 43, 44], 
    experiment_name="fast_sphere_opt", 
    output_dir="./results", 
    fitness="sphere:dim=10",
    genome_type="real",
    genome_length=10,
    bounds=(-5.0, 5.0),
    generations=10, 
    pop_size=32,
    elitism=2
)

for i in range(3):
    print(f"Run Status: {result.runs[i].status}")
    if result.runs[i].history:
        final_gen = result.runs[i].history[-1]
        print(f"Final Best Fitness: {final_gen['best_fitness']}")
```

## Integrating External Libraries

MalthusJAX's Composer integrates seamlessly with external optimization libraries. For instance, you can use the **EvoSAX** backend instead of native MalthusJAX operators by simply specifying `backend="evosax"` and selecting a strategy:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

result = composer.quick_run(
    fitness="bbob:fn_name=sphere,dim=10",
    backend="evosax",                     # Use the EvoSAX backend!
    evosax_strategy="CMA_ES",             # Select any EvoSAX strategy
    pop_size=64,
    generations=100,
    seeds=(42, 43)
)

print(result.aggregated_summary())
```

For advanced Quality-Diversity experiments, MalthusJAX provides native adapter builder functions (e.g., `build_qdax_engine`) that translate MAP-Elites grids seamlessly into the unified `Engine` protocol:

```python
import jax
import jax.numpy as jnp
import functools
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from malthusjax.composer.qdax_adapter import build_qdax_engine

# We wrap it in a mock object since build_qdax_engine looks for .scoring_function
class NativeEvaluator:
    def scoring_function(self, genotypes, random_key):
        # Sphere function negated for maximization
        fitnesses = -jnp.sum(jnp.square(genotypes), axis=-1)
        # Map the first two dimensions of the genotype into the [0, 1] range as descriptors
        descriptors = jnp.clip(genotypes[:, :2] / 10.0 + 0.5, 0.0, 1.0)
        return fitnesses, descriptors, {}

# 1. Prepare QDAX Centroids and Emitters
centroids = compute_cvt_centroids(
    num_descriptors=2, num_init_cvt_samples=10000, 
    num_centroids=100, minval=0.0, maxval=1.0, key=jax.random.PRNGKey(0)
)

emitter = MixingEmitter(
    mutation_fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1,
    variation_fn=lambda x1, x2, key: x1,
    variation_percentage=0.5,
    batch_size=50
)

# 2. Wrap them directly into a MalthusJAX engine
engine = build_qdax_engine(
    strategy_cls=MAPElites,
    emitter=emitter,
    metrics_function=functools.partial(default_qd_metrics, qd_offset=0.0),
    evaluator=NativeEvaluator(),
    eval_mode="native",
    init_variables=jnp.ones((50, 10)) * 5.0,
    centroids=centroids,
    pop_size=50,
    generations=100,
    history_metrics=["qd_score", "coverage"]
)

# 3. Run on the GPU at maximum speed
results = engine.run_once(jax.random.PRNGKey(42))
print(f"Final QD Score: {results['history'][-1]['qd_score']:.2f}")
print(f"Final Coverage: {results['history'][-1]['coverage']:.2f}")
```
