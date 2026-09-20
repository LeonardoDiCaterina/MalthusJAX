# `malthusjax.core.fitness` — Composable Evaluator Architecture & Reference

`malthusjax.core.fitness` defines how candidates are evaluated across benchmark problems, datasets, and dynamical simulations in **MalthusJAX**.

MalthusJAX separates candidate evaluation into a modular, 4-axis **Composable Evaluator Architecture** that eliminates the coupling between genome representations and problem environments.

---

## 1. The Composable Evaluator Paradigm

In monolithic evolutionary frameworks, evaluators tightly bundle genome decoding, dataset interactions, and fitness scoring into single bespoke classes. This creates a combinatorial explosion: evaluating a new genome encoding on an existing problem requires writing a brand-new evaluator class from scratch.

MalthusJAX decomposes candidate evaluation into four orthogonal axes:

$$\text{Evaluator} = \text{TaskShell}(\text{Environment} \times \text{Transform} \times \text{Interpreter} \times \text{Output})$$

```mermaid
graph LR
    Genome["Genome PyTree"] --> Transform["Transform (Genotype → Phenotype)"]
    Transform --> Interpreter["Interpreter (Decoding / Policy Call)"]
    Interpreter --> Environment["Environment (Task / Dataset / Rollout)"]
    Environment --> Output["Output (Scalar / MO / QD)"]
    Output --> Fitness["Fitness & State Transition"]
```

### Why This Matters:
- **Zero Combinatorial Explosion**: Any genome encoding (Real, Binary, Linear GP, TensorNEAT) can be plugged into any task (Analytical, Supervised, Reinforcement Learning).
- **Pure JAX Transformations**: Every stage is a pure function or `@struct.dataclass` PyTree compatible with `jax.vmap`, `jax.jit`, and `jax.lax.scan`.
- **Stateless & Type-Safe**: Problem datasets and static configurations are marked with `pytree_node=False` to prevent unnecessary JIT re-tracing.

---

## 2. The Three Composable Task Shells (`composable/evaluators.py`)

All evaluations are orchestrated through one of three composable task shells:

### 1. `OptimizationEvaluator`
- **Use Case**: Continuous benchmark functions, physics parameters, direct mathematical objectives, and combinatorial search.
- **Evaluation Flow**:
  1. `phenotype = transform(genome)`
  2. `candidate = interpreter(phenotype)`
  3. `raw_score = env.evaluate(candidate)`
  4. `fitness = output(raw_score)`

### 2. `SupervisedEvaluator`
- **Use Case**: Dataset-driven regression or classification tasks `(X, y)`.
- **Evaluation Flow**:
  1. `phenotype = transform(genome)`
  2. `predictor = interpreter(phenotype)`
  3. `y_pred = predictor(env.X)`
  4. `loss = env.compute_loss(y_pred, env.y)`
  5. `fitness = output(loss)`

### 3. `RLEvaluator`
- **Use Case**: Dynamic MDP episode rollouts (Brax, Gymnax, Jumanji).
- **Evaluation Flow**:
  1. `policy = interpreter(transform(genome))`
  2. Steps through `env.reset()` and `env.step()` using `jax.lax.scan` for `max_steps`
  3. Accumulates trajectory rewards
  4. `fitness = output(total_reward, trajectory_info)`

---

## 3. The Four Core Primitives (`composable/base.py`)

### Axis 1: Environment (`BaseEnvironment`)
Defines the task domain, dataset, or dynamical simulator:
- **`BaseOptimizationEnvironment`**: Exposes `evaluate(solution: chex.Array) -> chex.Numeric`.
- **`BaseSupervisedEnvironment`**: Stores dataset `data: Tuple[chex.Array, chex.Array] = (X, y)` marked with `pytree_node=False`.
- **`BaseRLEnvironment`**: Implements `reset(key)` and `step(state, action, key)`.

### Axis 2: Transform (`BaseTransform[G]`)
Translates the raw genetic representation into a decoded phenotype:
- **`IdentityTransform`**: Passes the genome through unchanged (default for direct encoding).
- **`TensorNeatTransform`**: Normalizes and structures variable-topology graph tensors.

### Axis 3: Interpreter (`BaseInterpreter[G]`)
Transforms the phenotype into an executable callable:
- **`IdentityInterpreter`**: Uses raw array values directly as the solution.
- **`LinearGPInterpreter`**: Executes Linear GP / MEP opcode sequences across input features.
- **`MLPInterpreter`**: Reshapes flat weight arrays into a neural network forward function.

### Axis 4: Output (`BaseOutput`)
Standardizes objective values and optimization direction:
- **`ScalarOutput(maximize=True/False)`**: Normalizes scalar fitness. Handles minimization/maximization sign conventions.
- **`MOOutput`**: Manages multi-objective score vectors for Pareto ranking (`MOEngine`).
- **`QDOutput`**: Extracts behavioral descriptors and fitness pairs for MAP-Elites archives (`QDEngine`).

---

## 4. Built-in Benchmark Evaluators

In addition to custom composable setups, MalthusJAX includes standard out-of-the-box evaluators:

### BBOB-JAX (`bbobax_evaluator.py`)
- High-performance, pure-JAX implementation of the 24 Black-Box Optimization Benchmarking (BBOB) functions.
- Integrates seamlessly with `BBOBAXEvaluator(config=BBOBAXConfig(fn_name="sphere", dim=10))`.
- Operates natively over `RealPopulation` using vectorized batch operations.

### Combinatorial Evaluators (`binary_evaluators.py`)
- **`BinarySumEvaluator`**: Classic OneMax problem for discrete search verification.
- **`KnapsackEvaluator`**: 0/1 Knapsack problem featuring a differentiable constraint violation penalty:
  $$\text{penalty} = \lambda \cdot \max(0, \text{weight} - \text{capacity})$$

### Linear GP Evaluator (`linear_gp_evaluator.py`)
- Fast linear instruction sequence evaluator for Multi-Expression Programming (MEP) with symbiotic best-node tracking.

---

## 5. Quickstart & Usage Examples

### Example 1: Composing a 3-Line Optimization Evaluator
```python
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from plugins.binary_sum_env import BinarySumEnv

# Compose: Environment + Transform + Interpreter + Output
evaluator = OptimizationEvaluator(
    env=BinarySumEnv(),
    transform=IdentityTransform(),
    interpreter=IdentityInterpreter(),
    output=ScalarOutput(maximize=True),
)

# Evaluate a batched population in parallel on the GPU
evaluated_pop = evaluator.evaluate_population(population)
```

### Example 2: Composing a Supervised Learning Evaluator
```python
import jax.numpy as jnp
from malthusjax.core.fitness.composable.evaluators import SupervisedEvaluator
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from plugins.linear_gp_interpreter import LinearGPInterpreter

# Dataset (X, y)
X = jnp.ones((100, 4))
y = jnp.zeros((100, 1))

evaluator = SupervisedEvaluator(
    env=CustomSupervisedEnv(data=(X, y)),
    transform=IdentityTransform(),
    interpreter=LinearGPInterpreter(),
    output=ScalarOutput(maximize=False),  # Minimize MSE
)
```

---

## 6. JAX & XLA Compilation Best Practices

To guarantee that evaluators fuse efficiently inside `jax.lax.scan` and `jax.jit` loops:

1. **Avoid Python Control Flow on Dynamic Data**:
   Never use Python `if/else` on evaluated candidate values or dynamic booleans. Use `jax.lax.select` or `jnp.where` so the compiler generates a single fused XLA computation graph.
2. **Mark Large or Static Data with `pytree_node=False`**:
   Datasets, lookup tables, and environment configs must be annotated with `struct.field(pytree_node=False)`. This ensures JAX treats them as static metadata rather than flattening them into active PyTree leaves.
3. **Keep Return Types Type-Stable**:
   Ensure `evaluate()` returns `chex.Numeric` or `chex.Array` with consistent shape and dtype across all execution paths.
