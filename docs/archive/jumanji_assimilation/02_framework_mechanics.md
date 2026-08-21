# Jumanji Framework Mechanics

This document breaks down the internal mechanics of the Jumanji framework and how its state and execution loop operate.

## 1. Initialization Protocol
- **Primary Object**: The environment is instantiated via `jumanji.make(env_name)`.
- **Reset Method**: `state, timestep = env.reset(rng_key)`
- **State and TimeStep**: Jumanji separates the internal environment state (`state`) from the user-facing observation and reward structure (`timestep`).
  - `state`: Internal JAX Pytree holding the ground truth (e.g., bin capacities, remaining items).
  - `timestep`: Follows the `dm_env.TimeStep` protocol. It contains `step_type`, `reward`, `discount`, and `observation`.

## 2. Execution Loop (Step Protocol)
- **Step Signature**: `state, timestep = env.step(state, action)`
- **Step Types**: Jumanji distinguishes between `FIRST`, `MID`, and `LAST` step types inside the `timestep`.
- **Auto-Reset**: Similar to standard JAX environments, Jumanji environments typically don't auto-reset without a wrapper.

## 3. Observations and Masking
- **Observation Space**: Often a complex dictionary (struct) containing both the actual observation and an `action_mask`.
- **Action Masking**: Because combinatorial tasks have dynamically changing legal actions, `timestep.observation.action_mask` is vital. Illegal actions lead to undefined behavior or immediate termination with a penalty.

## 4. PRNG Management
`rng_key` is used for initialization. Stochastic environments might use internal RNG keys within the `state`.
