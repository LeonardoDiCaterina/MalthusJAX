# Extension CLI Guide: Building MalthusJAX Projects

The MalthusJAX framework provides a powerful Extension CLI (`scripts/scaffold.py`) designed to help you rapidly prototype new algorithms, custom operators, and entire plugins from scratch.

This guide explains how to use the CLI to generate JAX-compliant templates, automatically write your testing boilerplate, and safely isolate your project components.

> [!TIP]
> The Extension CLI is not limited to any single "level" of the MalthusJAX API—it touches everything from defining fundamental `Genome` states to creating custom `Engine` pipelines.

---

## 1. Why Use the Scaffolding CLI?

When building evolutionary algorithms in JAX, strict adherence to `flax.struct.dataclass` constraints and array shapes is mandatory to ensure compilation (`@jax.jit`) and vectorization (`jax.vmap`) work safely. 

The CLI automates these constraints for you. By scaffolding components, you automatically get:
- **JAX-Compliant PyTrees:** All scaffolded classes are pre-decorated with `@struct.dataclass`.
- **Automatic Registration:** Components are automatically wired into the Composer catalog using `@register_...` decorators.
- **Boilerplate Compliance Testing:** The CLI writes `pytest` suites for your new components. These tests inherit from MalthusJAX's rigorous `ComplianceSuite` bases, automatically fuzz-testing your code against JAX constraints without you writing a single test case yourself!
- **Pre-Configured Unified Logging:** All scaffolded components import and instantiate `logger = get_logger(...)`, providing immediate diagnostic tracing without manual setup.


---

## 2. Command Reference

The Extension CLI is executed via `scripts/scaffold.py`. 

```bash
python scripts/scaffold.py -t <type> -n <ClassName> -k <registry_key>
```

### Arguments:
- `-t` / `--type` (Required): The type of component to scaffold. Valid options: `genome`, `population`, `selection`, `mutation`, `crossover`, `fitness`, `engine`, `interpreter`, `environment`, `adapter`.
- `-n` / `--name` (Required): The Python class name for your component (e.g., `QuantumMutation`).
- `-k` / `--key` (Required): The unique string key used to register the component in the composer catalog (e.g., `quantum_mutation`).
- `--impl-dir`: (Optional) The output directory for the implementation file. Defaults to `plugins`.
- `--test-dir`: (Optional) The output directory for the test file. Defaults to `tests/plugins`.

> [!IMPORTANT]
> The `--impl-dir` and `--test-dir` flags are crucial for **plugin isolation**. When creating an independent project (like a new algorithm), you should direct the scaffolding tools to output to your project's specific namespace rather than polluting the core `plugins/` directory.

---

## 3. Walkthrough: Building "Quantum GP" from Scratch

Let's walk through an example of building a brand-new experimental plugin called **Quantum Genetic Programming (QGP)**. We want this project isolated in its own folder structure: `plugins/qgp/`.

### Step 1: Scaffold the State Components
Evolutionary states require a `Genome` (representing the individual) and a `Population` (representing the batch).

Let's generate them, directing the CLI to our `qgp/` folder:

```bash
# Scaffold the Genome
python scripts/scaffold.py -t genome -n QuantumGenome -k quantum_genome \
  --impl-dir plugins/qgp/genomes --test-dir plugins/qgp/tests

# Scaffold the Population
python scripts/scaffold.py -t population -n QuantumPopulation -k quantum_population \
  --impl-dir plugins/qgp/populations --test-dir plugins/qgp/tests
```

### Step 2: Scaffold the Operators
Next, we need the genetic operators that manipulate the quantum states.

```bash
# Scaffold the Mutation Operator
python scripts/scaffold.py -t mutation -n QuantumMutation -k quantum_mutation \
  --impl-dir plugins/qgp/operators --test-dir plugins/qgp/tests

# Scaffold the Crossover Operator
python scripts/scaffold.py -t crossover -n QuantumCrossover -k quantum_crossover \
  --impl-dir plugins/qgp/operators --test-dir plugins/qgp/tests
```

### Step 3: Implement the Logic
The CLI will generate `plugins/qgp/operators/quantum_mutation.py` that looks like this:

```python
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseMutation
from malthusjax.composer import register_mutation

@register_mutation("quantum_mutation", override=True, compatible_genomes=["real", "continuous"])
@struct.dataclass
class QuantumMutation(BaseMutation):
    """A custom mutation operator."""
    
    mutation_rate: float = 0.1

    def _mutate_one(self, rng: jax.Array, genome: jax.Array) -> jax.Array:
        # TODO: Implement your mutation logic here
        return genome
```
All you need to do is fill in the `TODO` block! Because you inherit from `BaseMutation`, MalthusJAX will handle parallelizing this `_mutate_one` logic across the entire batch natively.

### Step 4: Run the Compliance Suites
When you scaffolded the components, the CLI also generated `plugins/qgp/tests/test_quantum_mutation.py`.

Without writing any extra test logic, simply run:
```bash
PYTHONPATH=. pytest plugins/qgp/tests/
```

MalthusJAX will aggressively validate your new component to ensure:
- It acts as a static JAX PyTree.
- It doesn't leak tracers or execute Python side-effects during JIT compilation.
- It complies with structural shape requirements.

### Step 5: Compose Your Project
Once your components pass compliance, they are automatically available in the MalthusJAX ecosystem! You can assemble your QGP project instantly using the Composer:

```python
from malthusjax.composer import Composer
import plugins.qgp.operators.quantum_mutation  # Ensure it registers

# MalthusJAX knows about your plugin via the "quantum_mutation" key!
config = {
    "engine": "genetic",
    "population_size": 100,
    "mutation": {
        "type": "quantum_mutation",
        "mutation_rate": 0.05
    }
}

engine, state = Composer.build(config)
```

By following this workflow, your custom projects remain cleanly isolated in their own domains, strictly compliant with JAX arrays, and immediately compatible with all higher-level MalthusJAX abstractions.

---

## 4. Scaffolding Composable Evaluator Components

With the new Composable Evaluator architecture, you evaluate genomes by composing an **Environment** and an **Interpreter**. You can scaffold these primitives directly!

```bash
# Scaffold an Interpreter (Genome Decoder)
python scripts/scaffold.py -t interpreter -n TransformerInterpreter -k transformer \
  --impl-dir plugins/interpreters --test-dir plugins/tests

# Scaffold an Environment (Dataset/Problem)
python scripts/scaffold.py -t environment -n JumanjiEnv -k jumanji \
  --impl-dir plugins/environments --test-dir plugins/tests
```

These generate boilerplate subclasses of `BaseInterpreter` and `BaseOptimizationEnvironment`, respectively, along with tests inheriting from `InterpreterComplianceSuite` and `EnvironmentComplianceSuite`.

---

## 5. Scaffolding Third-Party Framework Adapters

MalthusJAX can bridge external evolutionary frameworks (like Evosax or TensorNEAT) via **Universal Adapters**. The Scaffolding CLI can generate the boilerplate necessary to wrap external libraries!

```bash
# Scaffold an adapter for a framework called 'my_framework'
python scripts/scaffold.py -t adapter -n MyFrameworkAdapter -k my_framework \
  --impl-dir plugins/adapters --test-dir plugins/tests
```

This will generate an `AdapterComplianceSuite` test file and a source implementation looking like this:

```python
from malthusjax.composer.adapters import adapter, EvalMode
from malthusjax.composer.adapters.metrics import MetricSpec

ADAPTER_METRICS = [
    MetricSpec(name="best_fitness", source="best_fitness", is_objective_value=True),
    MetricSpec(name="mean_fitness", source="mean_fitness", is_objective_value=True),
]

@adapter(
    framework="my_framework",
    state_mapping={"init": "_adapter_init", "step": "_adapter_step"},
    eval_translators={EvalMode.NATIVE: lambda *args: None, EvalMode.MALTHUSJAX: lambda *args: None},
    metrics_catalog=ADAPTER_METRICS,
)
class MyFrameworkAdapter:
    """Universal Adapter for my_framework."""

    def _adapter_init(self, strategy, key, params, pop_init=None):
        # TODO: Initialize your framework's state here
        return None

    def _adapter_step(self, strategy, state, key, params, evaluator, eval_translator):
        # TODO: Implement a single step of your framework
        return state, {}
```

The `@adapter` decorator automatically injects MalthusJAX's `run_once` execution engine and bridges the fitness evaluation modes. The CLI also provides `pytest` compliance tests to ensure your wrapper seamlessly operates within the Composer ecosystem.
