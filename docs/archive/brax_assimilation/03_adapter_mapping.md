# Brax Adapter Mapping

This document details how MalthusJAX interfaces with Brax to evaluate continuous control policies.

## MalthusJAX Base Evaluator
As with Gymnax, the `BraxEvaluator` implements the `BaseEvaluator` protocol.

## Brax Mapping

### 1. Neural Network Policy (Phenotype)
Brax environments require continuous actions, typically normalized to `[-1.0, 1.0]`. The Flax policy must output continuous arrays with `action_size` matching the environment.
- **Policy Activation**: Ensure the final layer of the policy uses a `tanh` activation to bound actions, or scale them accordingly.

### 2. Rollout Function
The `lax.scan` loop for Brax is structurally similar to Gymnax but operates on the Brax `State` object.
- **Scan Over Steps**: The carry state must hold the Brax `State`, rather than a custom tuple of env variables.
- **Termination Masking**: Similar to Gymnax, if `done` is reached, subsequent rewards must be masked out to 0.

### 3. Population Vectorization
`jax.vmap` is extremely powerful with Brax. We can evaluate an entire population on a single accelerator device very quickly.
