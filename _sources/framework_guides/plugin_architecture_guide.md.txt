# MalthusJAX Plugin Architecture & Compatibility Guardrails

This guide explains how to extend MalthusJAX with custom engines, genomes, and operators *without* modifying the core repository, and how to ensure your custom components interact safely with the rest of the framework.

---

## 1. The Plugin Auto-Discovery System

MalthusJAX is designed to be highly modular. To prevent bloating the core repository with hyper-specialized domains (like Protein Folding, or specific logistic supply chain engines), MalthusJAX uses a two-tiered **Auto-Discovery System**.

Whenever you import the `malthusjax.composer` module, it automatically scans for extensions in two places:

### A. Local `plugins/` Directory (Rapid Prototyping)
For quick experiments or project-specific extensions, you can simply create a `plugins/` directory in your current working directory. 
Any `.py` file you drop into this folder will be automatically imported by the Composer. 

```text
my_experiment/
├── plugins/
│   ├── custom_operator.py
│   └── quantum_engine.py
└── run.py
```
As long as `custom_operator.py` uses the standard `@register_operator` decorators, those components will instantly be available in your TOML configurations or `Composer.quick_run()` calls.

### B. Python Entry Points (Standalone Packages)
For major extensions that you want to share with the community (e.g., `malthusjax-biology`), you should build a standalone Python package.

Instead of requiring users to manually import your package, you can declare your plugin using **Python Entry Points** in your `pyproject.toml`.

```toml
# In the pyproject.toml of your custom package
[project.entry-points."malthusjax.plugins"]
my_custom_domain = "malthusjax_biology.registry:register_all"
```
The moment a user runs `pip install malthusjax-biology`, the MalthusJAX Composer will automatically discover your package via `importlib.metadata` and load your engines/operators globally!

---

## 2. The Compatibility Guardrail System

Because MalthusJAX components are highly decoupled, users might accidentally define a TOML pipeline with impossible combinations (e.g., attempting to apply a continuous `gaussian` mutation to a discrete `binary` genome, or using a standard `tournament` selection on a `map_elites` engine).

Because JAX compiles the entire evolutionary loop as a single static graph, mismatches in PyTree shapes or types will result in massive, unreadable XLA compilation errors.

To solve this, MalthusJAX provides **Compatibility Guardrails** on its decorators.

### Defining Compatibility Metadata

When you register an operator or engine, you can (and should) explicitly declare which components it is compatible with using the `compatible_genomes` and `compatible_engines` arguments:

```python
from malthusjax.composer.decorators import register_operator

@register_operator(
    name="simulated_binary_crossover", 
    type="crossover", 
    compatible_genomes=["real", "continuous"],
    compatible_engines=["ga", "nsga2"]
)
def build_sbx(alpha=0.5):
    return SimulatedBinaryCrossover(alpha=alpha)
```

### The Factory Bouncer

When a user attempts to run a pipeline (either programmatically or via a TOML file), the Composer's Factory engine acts as a bouncer.

*Before* attempting to compile any JAX code, it inspects the resolved pipeline. If it detects a conflict, it will immediately halt the process and throw a clean, human-readable error:

```python
ValueError: Error: crossover is only compatible with genomes: ['real', 'continuous'], but you requested 'binary'.
```

By leveraging these metadata tags, you guarantee that your custom plugins are both easy to discover and completely safe to use within the broader MalthusJAX ecosystem.
