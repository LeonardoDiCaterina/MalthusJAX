# Jumanji Adapter Mapping

This document details how MalthusJAX interfaces with Jumanji for discrete, constrained policy evaluation.

## MalthusJAX Base Evaluator
The `JumanjiEvaluator` will implement the `BaseEvaluator` API, with special considerations for the `TimeStep` protocol.

## Jumanji Mapping

### 1. Neural Network Policy (Phenotype)
Policies for Jumanji must output discrete actions (e.g., using a Categorical distribution or argmax).
- **Masking**: The policy *must* accept the `action_mask` from `timestep.observation`. The network should set logits of illegal actions to `-inf` before sampling or taking the argmax.
- **Input Unpacking**: The observation is typically a struct, so the policy forward pass might need to specifically address `obs.grid`, `obs.capacity`, etc.

### 2. Rollout Function
The `lax.scan` loop must manage both `state` and `timestep`.
- **Scan Over Steps**: The carry state must hold `(state, timestep, cum_reward)`.
- **Termination Masking**: We use `timestep.last()` to detect termination. Subsequent rewards are masked.

### 3. Population Vectorization
`jax.vmap` handles Jumanji's nested structs (like `TimeStep` and `State`) out-of-the-box. We will vectorize over network parameters and environment seeds.
