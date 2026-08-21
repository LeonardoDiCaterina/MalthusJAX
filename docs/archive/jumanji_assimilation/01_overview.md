# Jumanji Assimilation Overview

## Purpose
The purpose of assimilating Jumanji into MalthusJAX is to provide benchmarking for combinatorial optimization and discrete reasoning tasks. Jumanji offers a suite of complex environments like Bin Packing, Job Shop Scheduling, and Traveling Salesperson, implemented natively in JAX.

## Key Capabilities
1. **Combinatorial Tasks**: Focuses on operations research and decision-making problems, distinct from the physics control of Brax.
2. **JAX-Native**: Fully compatible with JAX transformations (`jit`, `vmap`, `pmap`).
3. **Action Masking**: Inherent support for valid action masks, crucial for legal moves in constrained environments.

## Assimilation Goals
- **BaseEvaluator Integration**: Create a `JumanjiEvaluator` extending `BaseEvaluator`.
- **Timestep Management**: Bridge the dm_env-style `TimeStep` objects used by Jumanji.
- **Masking Integration**: Ensure neural network policies properly handle and mask invalid actions during the forward pass.

## References
- [Jumanji GitHub Repository](https://github.com/instadeepai/jumanji)
