# Level 4: The Composer 🎼

The **Composer** (`src/malthusjax/composer/`) is the fourth and highest abstraction layer in the MalthusJAX framework.

If Level 3 allows you to build and control a JAX Engine manually in Python, Level 4 is an orchestration engine that allows you to define an *entire experiment* via a dictionary or declarative `.toml` file and run it without writing a single line of execution code.

The Composer solves the primary pain points of neuroevolution research: configuring, wiring together, benchmarking, and tracking the results of complex multi-component evolutionary algorithms.

---

## 1. The Registry System (`decorators.py`)

The magic behind the Composer is a decentralized **Registry System**. Instead of maintaining massive switch-case statements inside the framework to instantiate classes, MalthusJAX uses factory decorators to map strings to builders.

> [!TIP]
> The decentralized registry means you can build custom modules completely isolated from the MalthusJAX source code, and inject them into the framework simply by importing your module before calling the Composer!

The core decorators include:
- `@register_genome(name="real")`
- `@register_fitness(name="sphere")`
- `@register_operator(name="gaussian_mutation", type="mutation")`
- `@register_engine(name="ga")`

When you decorate a factory function, it registers that tag globally:

```python
from malthusjax.composer.decorators import register_engine
from malthusjax.engine.genetic_fastengine import GeneticEngine

@register_engine("distance_matrix_ga")
def build_distance_engine(evaluator, selection, crossover, mutation, **kwargs):
    # This factory function tells the Composer HOW to build your custom Engine!
    return DistanceMatrixEngine(
        evaluator=evaluator,
        selection=selection,
        ...
    )
```

---

## 2. Dynamic Instantiation (`factory.py`)

When the Composer is given a configuration (e.g., specifying `"distance_matrix_ga"` as the engine type), it uses `factory.py` to dynamically instantiate and wire the components together in a specific topological order (Dependency Injection).

**The Pipeline Construction Order:**
1. **Genome Config**: Reads `genome_type` and shape bounds to construct the configuration.
2. **Evaluator**: Injects the genome config if necessary, constructs the fitness function.
3. **Operators**: Constructs Selection, Crossover, and Mutation operators.
4. **Engine**: Injects *all of the above* into the registered Engine factory to produce the final algorithm.

This completely abstracts away the massive boilerplate of wiring components together manually (as seen in Level 3).

---

## 3. The Pythonic Composer (Without TOML)

> [!NOTE]
> You don't *have* to use TOML files to take advantage of the Composer!

Because the Composer operates internally on raw Python dictionaries, you can define your pipeline configuration as a standard `dict` in your Python script or Jupyter Notebook.

By passing this dictionary directly into the Composer API, you can programmatically instantiate fully configured engines on the fly. 

```python
from malthusjax.composer.composer import Composer

composer = Composer.create_default()

# Run a fully wired experiment directly from Python dictionaries
result = composer.quick_run(
    experiment_name="programmatic_experiment",
    engine_type="ga",
    genome_type="real",
    fitness="sphere:dim=5",
    selection="tournament:tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.2",
    pop_size=32,
    generations=50,
    history_metrics=["best_fitness", "mean_fitness", "std_fitness", "distance_det"],
    seeds=[42, 100],  # Runs batched execution across multiple seeds!
)

print(result.aggregated_summary())
```

---

## 4. Declarative Experiments (TOML Parsing)

For reproducible research and large-scale benchmarking, the Composer can parse `.toml` configuration files. This allows researchers to version-control their experiments effortlessly.

You can define multiple parallel `[pipelines]` inside the same TOML file to run them sequentially, allowing for massive comparative benchmarks.

```toml
# experiment.toml
[global]
seeds = [42, 43, 44]
generations = 500
pop_size = 256
track_best = "FULL"

[pipelines.standard_ga]
engine_type = "ga"
genome_type = "real"
fitness = "sphere:dim=10"
selection = "tournament:tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.custom_ga]
engine_type = "distance_matrix_ga"
genome_type = "real"
fitness = "sphere:dim=10"
# ...
```
You simply load this using `Composer.from_toml("experiment.toml")`.

---

## 5. The Benchmark Runner (`benchmark_runner.py`)

The ultimate execution environment for a Composer pipeline is the `BenchmarkRunner`. Once the Composer dynamically constructs your engine, it hands it off to the BenchmarkRunner.

The BenchmarkRunner provides:
- **Strict Timing Profiling**: It enforces strict `jax.block_until_ready()` boundaries to accurately profile Wall-Clock time. Crucially, it separates **JIT Compilation Time (Warmup)** from actual **Execution Time**, providing fair benchmarking metrics across algorithms.
- **Data Collection**: It parses the heavy `jax.lax.scan` output histories based on the requested `history_metrics` parameter.
- **Serialization**: It automatically dumps the aggregated histories, metrics, Pareto fronts, and Quality-Diversity (QD) grids into serialized JSON/CSV files on disk for downstream plotting.

---

## 6. Interaction with Adapters

The Composer does not just orchestrate native MalthusJAX engines. It interacts natively with **Adapters** (see Level 4: Adapters Guide). 

Because Adapters wrap external libraries (like `evosax`, `qdax`, and `tensorneat`) into the unified `UniversalAdapterEngine` protocol, you can benchmark native MalthusJAX pipelines directly alongside EvoSAX pipelines inside the exact same `.toml` file or `composer.compare()` dictionary!

The BenchmarkRunner will execute them identically, time them identically, and aggregate their results into the same dataframes.
