---
name: New Problem Environment
about: Propose a new problem environment, dataset, or simulation task
title: '[ENV] '
labels: 'enhancement, core, fitness'
assignees: ''
---

## Description
Describe the problem environment or benchmark task you are proposing (e.g., a new combinatorial benchmark, continuous test suite, physics simulation, or supervised dataset).

## Environment Classification
Which abstract base class does this environment inherit from?
- [ ] **`BaseOptimizationEnvironment`**: Black-box objective function evaluation (`evaluate(solution) -> Numeric`).
- [ ] **`BaseSupervisedEnvironment`**: Dataset batching and loss computation (`X: Array, y: Array`, `loss_fn(preds, targets)`).
- [ ] **`BaseRLEnvironment`**: Episodic MDP / POMDP rollouts (`reset(key)`, `step(state, action, key)`, `obs_dim`, `action_dim`).

## Proposed Environment Schema
```python
# Proposed Environment Signature
@struct.dataclass
class MyProblemEnv(BaseOptimizationEnvironment):
    # static fields or PyTree parameters
    dim: int = struct.field(pytree_node=False, default=10)

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        # Objective computation
        ...
```

## Data Source & Parameterization
How is the problem data generated or loaded?
- [ ] Synthetic on-the-fly generation (e.g., using random seed)
- [ ] Static array / distance matrix embedded in dataclass
- [ ] External data loader / Option C data ID (`[data.<id>]` in TOML)

## Composable Evaluator Pairing
How will this environment pair with other composable primitives?
- **Expected Interpreter**: (e.g., `IdentityInterpreter`, `MLPInterpreter`, `LinearGPInterpreter`)
- **Default Output Mode**: [ ] `ScalarOutput` / [ ] `MOOutput` / [ ] `QDOutput`
- **Natural Direction**: [ ] Minimization / [ ] Maximization

## JAX JIT & Accelerator Feasibility
Is the environment execution 100% pure JAX (suitable for `jax.jit` and `jax.vmap`)?
- [ ] Yes (pure JAX operations only)
- [ ] No (requires host callbacks or CPU synchronization)
