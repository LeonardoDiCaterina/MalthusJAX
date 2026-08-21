# Gymnax Assimilation Overview

## Purpose
The purpose of assimilating Gymnax into MalthusJAX is to provide rapid, hardware-accelerated benchmarking for reinforcement learning agents across a suite of classic control, bsuite, and MinAtar environments. Gymnax is the JAX-native equivalent of OpenAI Gym, making it an ideal choice for testing evolutionary algorithms on standard control tasks.

## Key Capabilities
1. **JAX-Native Execution**: Environments are written entirely in JAX, enabling seamless `jax.jit` compilation and massive parallelization via `jax.vmap`.
2. **Standardized API**: Follows a functional approach similar to OpenAI Gym, mapping well to existing RL evaluation patterns.
3. **Diverse Suite**: Includes Classic Control (e.g., CartPole, Pendulum), BSuite (behavioral tasks), and MinAtar (miniature Atari games).

## Assimilation Goals
- **BaseEvaluator Integration**: Create a `GymnaxEvaluator` that extends `BaseEvaluator` from MalthusJAX.
- **Batched Rollouts**: Leverage `jax.vmap` internally to evaluate a population of policies across multiple environment seeds simultaneously.
- **Neural Network Policies**: Bridge the gap between evolutionary generated parameters (genotypes) and the forward pass of Flax-based neural networks (phenotypes) to produce actions.

## References
- [Gymnax GitHub Repository](https://github.com/RobertTLange/gymnax)
