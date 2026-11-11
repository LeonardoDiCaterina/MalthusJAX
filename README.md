# MalthusJAX 🧬⚡

[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[Coverage scope and validation](docs/audit_remediation/P11.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**From Raw JAX PyTrees to Declarative TOML Sweeps: A Multi-Level Evolutionary Framework.**

MalthusJAX is not just a high-level CLI tool for running experiments. It is a deeply modular, 4-tier evolutionary computation framework where every layer is designed to run synchronously on GPUs/TPUs using `jax.lax.scan`. 

Whether you want to declare a massive multi-seed parameter sweep via a TOML file, or you just want to rip out a blazingly fast, vectorized Simulated Binary Crossover (SBX) operator to plug into your own custom RL training loop, MalthusJAX scales to your exact level of abstraction.

*No Python loop overhead. No recompilation between generations. Just fast, scalable JAX state transitions.*

---

## 🧅 The Multi-Level API: Choose Your Abstraction

MalthusJAX is designed so you can enter the framework at the exact depth your project requires.

### [Level 4: The Composer](src/malthusjax/composer/README.md) (Highest Abstraction)
- **What it is**: The `Composer` API and the unified `mjax` CLI.
- **Use case**: You want to define declarative TOML experiments, resolve plugins dynamically, run statistical parity testing, and launch massive multi-seed sweeps without writing boilerplate.
- **Example**: `mjax run experiment.toml`

### [Level 3: The Engine](src/malthusjax/engine/README.md)
- **What it is**: Core evolutionary orchestration like `GeneticEngine` (aliased as `GeneticFastEngine`), `MOEngine`, `MapElitesEngine`, and `BaseIslandModel`.
- **Use case**: You want full control over the `jax.lax.scan` loop (e.g., to step custom RL environments or implement unique logging), but you want MalthusJAX to handle the 5-phase generation logic (Entropy → Selection → Crossover → Mutation → Evaluation).
- **Example**: `next_state, output = engine.step(state)`

### [Level 2: The Operators](src/malthusjax/operators/README.md)
- **What it is**: Pre-built, vectorized genetic operators (`BaseMutation`, `BaseCrossover`, `BaseSelection`).
- **Use case**: You already have a custom training loop built in JAX, and you simply need a highly optimized, GPU-accelerated crossover or mutation function to drop in.
- **Example**: `mutated_pop = mutation(keys, population, config)`

### [Level 1: The Core](src/malthusjax/core/README.md)
- **What it is**: Low-level foundational PyTrees (`BaseGenome`, `BasePopulation`, `malthusjax.core.random`) and the new 4-axis Composable Evaluator architecture (`Environment × Transform × Interpreter × Output`).
- **Use case**: You need deterministic PRNG key management, specialized genome representations (e.g., TensorNEAT graphs or bitstrings), or custom evaluation task composition.
- **Example**: `keys = malthusjax.core.random.split(key, 5)`

---

## ⚡ Quickstart (Level 4: The Composer)

> 💡 **Interactive Tutorial**: Open **[`notebooks/00_quickstart_in_5_minutes.ipynb`](notebooks/00_quickstart_in_5_minutes.ipynb)** for a hands-on, 5-minute Jupyter walkthrough featuring convergence plots, custom operators, and algorithm comparisons.

Get your first hardware-accelerated evolutionary pipeline running in under 5 lines of code:

### Option A: Interactive Exploratory Run
```python
import malthusjax as mj

composer = mj.Composer.create_default()
result = composer.quick_run(
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
)
print(result.aggregated_summary())
```

### Option B: Production Typed Configuration
```python
from malthusjax.composer.experiment_config import (
    ExperimentConfig, PopulationConfig, ExecutionConfig, MalthusJAXBackendConfig
)
from malthusjax.composer.engine_factory import EngineFactory
from malthusjax.benchmarking.runner import BenchmarkRunner

# 1. Type-safe, immutable config tree
config = ExperimentConfig(
    population=PopulationConfig(size=100, genome_length=10),
    execution=ExecutionConfig(generations=200, seeds=(42, 43, 44)),
    backend=MalthusJAXBackendConfig(
        fitness="sphere:dim=10",
        selection="tournament:num_selections=64,tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.1",
        elitism=2,
    ),
)
config.validate()

# 2. Build via single factory boundary & execute
engine = EngineFactory.build(config)
runner = BenchmarkRunner(engine=engine, experiment_name=config.output.experiment_name)
result = runner.run(seeds=config.execution.seeds)
print(result.aggregated_summary())
```

---

## 🧠 Architecture & "Why JAX?"

Evolutionary algorithms look embarrassingly parallel, but naive implementations repeatedly cross the Python/JAX boundary every generation, resulting in massive host-to-device communication overhead.

MalthusJAX models an entire evolutionary generation as a **pure PyTree state transition**, compiling the *entire multi-generation loop* into a single XLA program using `jax.lax.scan`:

$$\text{State}_t \xrightarrow{\text{entropy + selection + reproduction + evaluation}} \text{State}_{t+1}$$

```mermaid
graph TD
    subgraph composer_layer["Composer Layer (Level 4)"]
        cli["Unified mjax CLI"] --> config["ExperimentConfig / TOML"]
        config --> factory["EngineFactory Boundary"]
        factory --> registry["BackendRegistry"]
        registry --> runner["Composer / BenchmarkRunner"]
    end

    subgraph engine_layer["Engine Layer (Level 3)"]
        runner --> base_engine["GeneticEngine / GeneticFastEngine"]
        runner --> mo_engine["MOEngine (NSGA-II)"]
        runner --> island_meta["BaseIslandModel"]
        runner --> adapters["Universal Adapters (EvoSAX, QDAX)"]
    end

    subgraph operators_layer["Operators Layer (Level 2)"]
        base_engine --> selection["Selection Operators"]
        base_engine --> crossover["Crossover Operators"]
        base_engine --> mutation["Mutation Operators"]
    end
    
    subgraph core_layer["Core State Layer (Level 1)"]
        selection --> core["BasePopulation & Genomes (PyTrees)"]
        crossover --> core
        mutation --> core
    end
```

### Benchmark results

The full Phase 1 GPU study contains **64,800 runs** over 108 settings.
It compares direct upstream execution, the MalthusJAX adapter, and the native
mimic for EvoSAX SimpleGA and QDax MAP-Elites. The grid crosses six BBOB
functions, two instance seeds, dimensions 10/20/50, and generation budgets
50/150/500 at population 128, with 100 matched seeds per setting.

| Native vs. adapter | EvoSAX SimpleGA | QDax MAP-Elites |
|---|---:|---:|
| Pooled log₁₀ objective error difference | +0.00192 | −0.00502 |
| 90% CI lower bound | −0.00201 | −0.01210 |
| 90% CI upper bound | +0.00585 | +0.00206 |
| Settings meeting the ±0.05 equivalence bound | 85/108 | 53/108 |
| Geometric mean steady GPU speedup | 1.8175× | 0.9964× |
| 95% CI lower bound for speedup | 1.8169× | 0.9961× |
| 95% CI upper bound for speedup | 1.8180× | 0.9966× |

All six pooled objective contrasts met the equivalence bound. Adapter versus
direct upstream met it in all 108 settings for both libraries. QDax native
versus adapter archive coverage met the ±5 percentage point bound in all 108
settings. The frozen relative QD-score test was inconclusive.

Speedup is reference time divided by contender time, using the median of five
synchronized steady executions per run. Compilation and warm-up were measured
separately. The intervals use paired estimates clustered by seed.
[Study protocol](configs/publication/part1_confirmatory.json) ·
[Artifact specification](docs/part1_confirmatory_v2_artifact_spec.md)

> [!NOTE]
> **Optimization Direction Convention & Benchmark Re-run Advisory**:
> MalthusJAX enforces a canonical **lower-is-better (minimization)** convention across all engines and evaluators. Evaluators constructed with `maximize=True` negate raw objectives (`-score`), while engines minimize uniformly. Benchmark results involving `BinarySumEvaluator` and `KnapsackEvaluator` (under both `maximize=True` and `maximize=False`) require regeneration following the standardization of evaluation signs.
>
> *Migration notice*: If you were using `maximize=False` on `BinarySumEvaluator` or `KnapsackEvaluator` expecting it to find good solutions, switch to `maximize=True` — the old behavior was accidental due to inverted proxy encodings.

---

## 🛠️ The Ecosystem & Integrations

MalthusJAX is built to be extensible. You can write custom plugins (like a new mutation operator) and seamlessly integrate them into the ecosystem.

### The Scaffolding CLI & Standalone Compliance Suite
Writing custom operators in JAX can be tricky (e.g., getting the `vmap` axes and PyTree shapes exactly right). MalthusJAX provides a built-in scaffolding CLI that instantly generates boilerplate code for you:

```bash
python scripts/scaffold.py -t mutation -n QuantumMutation -k quantum_mutation
```
This generates your plugin *and* a `pytest` file that inherits from our **Standalone Compliance Suite**. The compliance suite automatically runs rigorous `jax.jit` and shape-contract tests on your plugin to ensure it will work flawlessly inside a complex MalthusJAX compiled loop. For complete usage across all 12 component types, see the [Extension CLI Guide](docs/source/extension_cli_guide.md).

For declarative experiment definitions, multi-operator ablations, and automated benchmarking suites, see the [TOML Configuration Guide](docs/source/toml_guide.md).

### External Library Integrations
MalthusJAX acts as a universal bridge, allowing you to natively compile and benchmark external libraries alongside MalthusJAX strategies:
- **EvoSAX:** Run any EvoSAX strategy natively through our `Composer`.
- **QDAX (Quality-Diversity):** Native adapter builders translate MAP-Elites grids into the unified Engine protocol.
- **TensorNEAT:** Evolve complex neural network topologies natively on accelerators.
- **Hardware-Accelerated RL:** Deep native integrations with **Gymnax**, **Brax**, and **Jumanji** to evolve neural network policies at millions of frames per second.

---

## 🚀 Installation & Deployment

### Local Installation
```bash
# Clone the repository
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX

# Install in development mode with all dependencies
make install-dev
```
*Requires Python 3.12+ and JAX 0.11–0.12 (see validation scope).*

### Docker Deployment
MalthusJAX includes a robust, multi-stage `Dockerfile` to simplify deployment on shared GPU clusters. Isolate optional dependencies (`qdax`, `evosax`, `tensorneat`, `rl`) during the build:

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --build-arg EXTRAS="[cuda12,qdax,rl]" \
  -t malthusjax:latest .
```

---

## 🤝 Contributing

We are actively building a comprehensive "rack" of Genetic Operators and Evolutionary Engines! Please read our **[CONTRIBUTING.md](CONTRIBUTING.md)** for our Golden Rules (`jax.jit` / `flax.struct` compliance) and instructions on how to use the built-in **Scaffolding CLI**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.