# Gymnax Adapter Mapping

This document details how MalthusJAX interfaces with Gymnax to evaluate neural network policies using evolutionary strategies.

## MalthusJAX Base Evaluator
MalthusJAX defines a `BaseEvaluator` that expects two primary methods:
- `evaluate(rng, params)`: Evaluates a single set of parameters.
- `evaluate_population(rng, pop_params)`: Evaluates a batch of parameters.

## Gymnax Mapping

### 1. Neural Network Policy (Phenotype)
Evolutionary algorithms maintain a population of parameter arrays (genotypes). These parameters must be injected into a Flax `nn.Module` to compute actions.
- **Policy Signature**: `action = policy.apply(params, obs)`
- **Genotype to Phenotype**: The evaluator will take the raw arrays from the optimizer (e.g., NSGA-II) and treat them as the weights of the Flax module.

### 2. Rollout Function
The core mechanism requires mapping the episodic rollout to a JAX `lax.scan` for efficiency.
- **Scan Over Steps**: A helper function (e.g., `_rollout_episode`) uses `jax.lax.scan` to execute the environment loop up to `max_steps`.
- **Termination Masking**: Since `jax.lax.scan` has a fixed length, rewards accumulated after `done == True` must be masked out to 0.0.

### 3. Population Vectorization
To evaluate an entire population simultaneously:
- **vmap over population**: `jax.vmap(_rollout_episode, in_axes=(None, 0))` maps the evaluator across the population parameters.
- **vmap over episodes**: To evaluate robustly across multiple seeds, `jax.vmap` is also applied over a batch of initial `rng_keys`.

## Expected Evaluator Signatures
- **Constructor**: `__init__(self, env_name: str, policy: nn.Module, max_steps: int = 1000)`
- **evaluate**: `def evaluate(self, rng: jnp.ndarray, params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]` returns fitness, output state, time.
