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
