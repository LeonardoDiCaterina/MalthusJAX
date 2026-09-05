# Extending Fitness Evaluators in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers who need to evaluate genomes against custom physics simulators, bespoke RL environments, complex supervised learning metrics, or Quality-Diversity (QD) objectives.

In MalthusJAX, the evaluation pipeline is designed to be fully JIT-compilable and vectorized. A **Fitness Evaluator** defines the logic for evaluating a *single* genome. MalthusJAX automatically "lifts" this logic to evaluate the entire population in parallel using `jax.vmap`.

This guide covers how to write, configure, and register a custom fitness evaluator.

---

## 1. The Core Architecture: `BaseEvaluator`

All evaluators inherit from `BaseEvaluator[G, C, D]` (Genome type, Config type, Data type), located in `src/malthusjax/core/fitness/base.py`.

### The Three Pillars of an Evaluator:
1. **`evaluate(self, genome: G) -> chex.Numeric`**: The only mathematical method you *must* implement. It takes an unbatched, single PyTree genome and returns a scalar fitness score.
2. **`config` (Static)**: A configuration dataclass (inheriting from `BaseEvaluatorConfig`) holding static hyper-parameters like `maximize=True` or `loss_fn="mse"`. Because it is marked as static, JAX will not attempt to trace or differentiate it.
3. **`data` (Static/Constant)**: The evaluation data. For supervised learning, this is the `(X, y)` batch. For RL, it might be the environment parameters. This is held constant across the population dimension during `vmap`.

---

## 2. Writing a Custom Evaluator (Example: Custom Physics)

Suppose we have a JAX-compatible physics simulation function `simulate_trajectory(params, initial_state)` and we want to score genomes based on how far a simulated object travels.

### Step 1: Define the Config
```python
from flax import struct
from malthusjax.core.fitness.base import BaseEvaluatorConfig

@struct.dataclass
class PhysicsEvaluatorConfig(BaseEvaluatorConfig):
    # BaseEvaluatorConfig already provides: maximize, batch_size, loss_function
    # Let's add custom static fields:
    simulation_steps: int = struct.field(pytree_node=False, default=100)
    gravity: float = struct.field(pytree_node=False, default=9.81)
```

### Step 2: Create the Evaluator Class
```python
from typing import Any
import chex
from malthusjax.core.fitness.base import BaseEvaluator

@struct.dataclass
class PhysicsEvaluator(BaseEvaluator[Any, PhysicsEvaluatorConfig, chex.Array]):
    """
    Evaluates a genome by running a custom JAX physics simulation.
    Here, `data` represents the initial state of the simulation.
    """

    def evaluate(self, genome: Any) -> chex.Numeric:
        """
        Evaluate a SINGLE genome.
        Because MalthusJAX uses SoA batching, this method will automatically
        be vmapped across the entire population!
        """
        initial_state = self.data # e.g. [x, y, vx, vy]
        
        # 1. Extract the parameters from the genome
        # (Assuming the genome has a 'values' array)
        params = genome.values 
        
        # 2. Run the simulation (this must be pure JAX!)
        # Assuming simulate_trajectory is some jax.lax.scan based function
        final_state = simulate_trajectory(
            params, 
            initial_state, 
            steps=self.config.simulation_steps, 
            g=self.config.gravity
        )
        
        # 3. Calculate distance traveled (Fitness)
        distance = jnp.linalg.norm(final_state[:2] - initial_state[:2])
        
        # 4. Return scalar fitness. 
        # Note: If config.maximize is False (default), the GeneticEngine
        # assumes lower is better. Since we want higher distance, we must 
        # ensure maximize=True in the TOML, or return -distance.
        return distance
```

> [!IMPORTANT]
> **No Batch Dimensions Here!**
> Inside `evaluate()`, you are operating on a *single* genome and the *shared* `self.data`. Do not write `vmap` logic over the population size here; `BaseEvaluator.evaluate_population` handles the population vectorization automatically.

---

## 3. The Core Philosophy: Why `evaluate_population`?

While you only ever write the single-genome `evaluate()` logic, it's crucial to understand why MalthusJAX orchestrates this through the `evaluate_population()` wrapper. 

Because `evaluate_population` processes the entire batch at once, **it has access to the fitness scores of the entire population simultaneously**. 

This access justifies a massive architectural responsibility: **the Evaluator (and the resulting Population object) is the absolute source of truth for Elitism and ranking.** 

By having the Evaluator return a fully evaluated `Population`, the engine doesn't have to guess how to rank individuals. If the algorithm requires complex elitism (like Non-Dominated Sorting for Pareto fronts, or Novelty Archives for QD), the Evaluator has all the global information it needs right there to compute and pack that elitism info directly into the returned `Population` object (e.g., via `MOPopulation.from_evaluated`).

---

## 4. Interaction with Predictors and `unflatten_fn`

In many tasks (like RL or Sklearn regression), the genome represents the weights of a Neural Network (a `Predictor` or `Policy`). 
Because the Evaluator doesn't know the exact PyTree structure of the genome (e.g. `RealGenome` vs `MaskedGenome`), it delegates the weight-parsing back to the genome using `unflatten_fn()`.

If your Evaluator relies on a Predictor, the `evaluate` method usually looks like this:
```python
def evaluate(self, genome: Any) -> chex.Numeric:
    # 1. Ask the genome to unflatten its flat array into the PyTree 
    # structure expected by the Predictor.
    structured_params = genome.unflatten_fn()
    
    # 2. Use the predictor
    predictions = self.config.predict_fn(structured_params, self.data[0])
    
    # ... compute loss ...
```

---

## 5. Extending for Quality Diversity (QD)

If you are using MAP-Elites or Novelty Search, your evaluator must return both a fitness score AND a Behavioral Descriptor.

Instead of `BaseEvaluator`, subclass `BaseQDEvaluator` (found in `src/malthusjax/core/fitness/qd_evaluator.py`).

```python
from malthusjax.core.fitness.qd_evaluator import BaseQDEvaluator

@struct.dataclass
class QDPhysicsEvaluator(BaseQDEvaluator[Any, PhysicsEvaluatorConfig, chex.Array]):

    def evaluate(self, genome: Any) -> tuple[chex.Numeric, chex.Array]:
        """
        Notice the return signature! It now requires a tuple of (fitness, descriptor).
        """
        final_state = simulate_trajectory(genome.values, self.data)
        
        fitness = jnp.linalg.norm(final_state[:2] - self.data[:2])
        
        # Descriptor: The final X,Y coordinates where the object landed
        descriptor = final_state[:2] 
        
        return fitness, descriptor
```

When `evaluate_population` runs on a `BaseQDEvaluator`, it intercepts the returned tuple. It assigns the `fitness` to `population.fitness`, and seamlessly packs the `descriptor` batch into `population.info["descriptors"]`.

---

## 6. Extending for Multi-Objective (MO) Evolution

If your problem has multiple competing objectives (e.g., maximizing accuracy while minimizing model size), you should subclass `BaseMOEvaluator` (found in `src/malthusjax/core/fitness/mo/evaluator.py`).

```python
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator

@struct.dataclass
class MOPhysicsEvaluator(BaseMOEvaluator[Any, PhysicsEvaluatorConfig, chex.Array]):

    def evaluate(self, genome: Any) -> chex.Array:
        """
        Return a chex.Array containing multiple fitness scores.
        """
        final_state, energy_used = simulate_trajectory(genome.values, self.data)
        
        distance = jnp.linalg.norm(final_state[:2] - self.data[:2])
        
        # Return a vector of objectives
        return jnp.array([distance, -energy_used])
```

### The MO Upgrade Magic
When `BaseMOEvaluator.evaluate_population` is called, it doesn't just return a standard `BasePopulation`. It intercepts the matrix of fitness scores and calls `MOPopulation.from_evaluated()`. 

This instantly upgrades your population to an `MOPopulation`, which automatically calculates the **Pareto Ranks** and **Crowding Distances** for the entire batch via Non-Dominated Sorting. Your downstream Multi-Objective genetic engine (like NSGA-II) can then immediately use these ranks for tournament selection!

---

## 7. Registering the Custom Evaluator

To make your evaluator available to the `Composer` via TOML configurations, you must register it using the `@register_fitness` factory decorator.

Create a new file, or place this at the bottom of your evaluator file:

```python
from malthusjax.composer.decorators import register_fitness
from malthusjax.core.fitness.base import BaseEvaluator

@register_fitness("custom_physics")
def build_custom_physics_evaluator(**kwargs) -> BaseEvaluator:
    """Factory function for the Composer."""
    
    # 1. Parse kwargs from TOML
    config = PhysicsEvaluatorConfig(
        maximize=kwargs.get("maximize", True),
        simulation_steps=kwargs.get("simulation_steps", 100),
        gravity=kwargs.get("gravity", 9.81)
    )
    
    # 2. Setup the static data
    initial_state = jnp.array([0.0, 10.0, 5.0, 0.0])
    
    return PhysicsEvaluator(config=config, data=initial_state)
```

---

## 8. Using it in TOML

With the evaluator registered, you can invoke it directly from your experiment configuration!

```toml
[pipelines.my_pipeline]
engine_type = "ga"

[pipelines.my_pipeline.fitness]
type = "custom_physics"
maximize = true
simulation_steps = 500
gravity = 9.81
```

> [!WARNING]
> **Data Loading**: In a real-world scenario, your factory function might need to load data from disk (like an Sklearn dataset). Do this *inside* the factory function (`build_custom_physics_evaluator`), convert it to `jnp.array`, and pass it into the `PhysicsEvaluator` constructor as `data`. Never perform file I/O inside the `evaluate()` method!
