# Extending Fitness Evaluators in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers who need to evaluate genomes against custom physics simulators, bespoke RL environments, complex supervised learning metrics, or Quality-Diversity (QD) objectives.

MalthusJAX offers two ways to build fitness evaluators:
1. **Composable Evaluators (Recommended)**: The new three-axis decomposition (TaskType × Environment × Interpreter). Use this for any task involving Neural Networks, Genetic Programming, or complex environments (RL/Supervised).
2. **Legacy Monolithic Evaluators**: The original `BaseEvaluator` approach. Best for very simple, bespoke objective functions that don't need decoding or environments.

---

## Part A: Composable Evaluators (Recommended)

In the composable architecture, you don't write a monolithic evaluator. Instead, you compose primitives according to this formula:

`Evaluator = Problem(TaskType × Environment × Interpreter) × Output`

You write custom extensions by subclassing the primitives:

### 1. Writing a Custom Environment

An Environment provides the problem instance or dataset. It defines *what* is being solved.

- Inherit from `BaseSupervisedEnvironment` for datasets `(X, y)`.
- Inherit from `BaseRLEnvironment` for episode rollouts.
- Inherit from `BaseOptimizationEnvironment` for direct data-defined objectives (like TSP, Knapsack, or bespoke physics sims).

Example: A custom physics environment.
```python
import chex
import jax.numpy as jnp
from flax import struct
from malthusjax.core.fitness.composable.base import BaseOptimizationEnvironment

@struct.dataclass
class PhysicsEnv(BaseOptimizationEnvironment):
    """A custom physics environment."""
    
    # Store initial state as a static field (not traced by JAX)
    initial_state: chex.Array = struct.field(pytree_node=False)
    
    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a solution vector against the environment."""
        # Custom pure-JAX simulation logic
        final_state = simulate_trajectory(solution, self.initial_state)
        distance = jnp.linalg.norm(final_state[:2] - self.initial_state[:2])
        return distance
```

### 2. Writing a Custom Interpreter (Decoder)

If your genome doesn't represent the direct solution, you need an Interpreter to decode it into a callable (like an MLP or a GP graph).

For full details on Interpreters, see the [Interpreter Guide](interpreter_guide).

### 3. Composing the Evaluator

Once you have your custom environment and interpreter, compose them with an Output mode:

```python
from malthusjax.core.fitness.composable import OptimizationEvaluator, IdentityInterpreter, ScalarOutput

env = PhysicsEnv(initial_state=jnp.array([0.0, 10.0, 5.0, 0.0]))
interp = IdentityInterpreter() # Genome is the raw parameter vector
output = ScalarOutput(maximize=True)

# The orchestrator handles the vmap over the population automatically
evaluator = OptimizationEvaluator(env=env, interpreter=interp, output=output)
```

### Using it in TOML

To use this composition in a TOML config, use the new `[fitness]` structured subsection:

```toml
[pipelines.my_physics]
engine_type = "ga"
genome_type = "real"
genome_length = 10

[pipelines.my_physics.fitness]
# Assuming PhysicsEnv is registered in Composer
env = "custom_physics"
env.initial_state = [0.0, 10.0, 5.0, 0.0]

interpreter = "identity"

output = "scalar"
output.maximize = true
```

---

## Part B: Legacy Monolithic Evaluators

> [!WARNING]
> While supported, the monolithic approach suffers from a combinatorial explosion if you want to support multiple genome types (e.g., you'd have to write `PhysicsMEPEvaluator`, `PhysicsCGPEvaluator`, etc.). Prefer Part A when possible.

If you just have a simple function and want to write the entire evaluation loop yourself, inherit from `BaseEvaluator[G, C, D]` in `src/malthusjax/core/fitness/base.py`.

### 1. Define the Config
```python
from flax import struct
from malthusjax.core.fitness.base import BaseEvaluatorConfig

@struct.dataclass
class LegacyPhysicsConfig(BaseEvaluatorConfig):
    simulation_steps: int = struct.field(pytree_node=False, default=100)
    gravity: float = struct.field(pytree_node=False, default=9.81)
```

### 2. Create the Evaluator Class
```python
from typing import Any
import chex
from malthusjax.core.fitness.base import BaseEvaluator

@struct.dataclass
class LegacyPhysicsEvaluator(BaseEvaluator[Any, LegacyPhysicsConfig, chex.Array]):
    
    def evaluate(self, genome: Any) -> chex.Numeric:
        """Evaluate a SINGLE genome. This gets automatically vmapped."""
        initial_state = self.data
        params = genome.values 
        
        final_state = simulate_trajectory(
            params, 
            initial_state, 
            steps=self.config.simulation_steps, 
            g=self.config.gravity
        )
        return jnp.linalg.norm(final_state[:2] - initial_state[:2])
```

> [!IMPORTANT]
> **No Batch Dimensions Here!**
> Inside `evaluate()`, you are operating on a *single* genome and the *shared* `self.data`. Do not write `vmap` logic over the population size here; `BaseEvaluator.evaluate_population` handles the population vectorization automatically.

### 3. QD and Multi-Objective (Legacy)

In the legacy architecture, you had to subclass specific evaluators:
- For QD, subclass `BaseQDEvaluator` and return `(fitness, descriptor)`.
- For MO, subclass `BaseMOEvaluator` and return a vector `chex.Array`.

In the new Composable architecture, you simply change the Output mode (e.g. `QDOutput(descriptor_fn=...)` or `MOOutput(n_objectives=...)`).

### 4. Registering the Legacy Evaluator

```python
from malthusjax.composer.decorators import register_fitness

@register_fitness("legacy_physics")
def build_legacy_physics(**kwargs) -> BaseEvaluator:
    config = LegacyPhysicsConfig(
        maximize=kwargs.get("maximize", True),
        simulation_steps=kwargs.get("simulation_steps", 100)
    )
    initial_state = jnp.array([0.0, 10.0, 5.0, 0.0])
    return LegacyPhysicsEvaluator(config=config, data=initial_state)
```

Used in TOML via the flat string syntax:
```toml
fitness = "legacy_physics:maximize=true,simulation_steps=500"
```
