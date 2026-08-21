# Brax Framework Mechanics

This document breaks down the internal mechanics of the Brax framework and how its state and execution loop operate.

## 1. Initialization Protocol
- **Primary Object**: The environment is instantiated via `brax.envs.create(env_name)`.
- **Reset Method**: `state = env.reset(rng_key)`
- **State Object**: The Brax `State` object is much more complex than Gymnax. It contains:
  - `obs`: The observation array.
  - `reward`: The reward for the step.
  - `done`: The termination flag.
  - `metrics`: A dictionary of environment-specific metrics.
  - `info`: Additional info.
  - `qp`, `pipeline_state`, etc.: The underlying physics state (positions, quaternions, velocities).

## 2. Execution Loop (Step Protocol)
- **Step Signature**: `next_state = env.step(state, action)`
  - Notice that `rng_key` is not explicitly passed to `step` in standard Brax environments (it is managed internally within the state if needed for noise, though Brax envs are largely deterministic given the initial seed).
  - The return is a new `State` object containing the updated `obs`, `reward`, and `done`.
- **Auto-Reset**: Standard Brax environments often *do not* auto-reset by default unless wrapped. We must be mindful of this when running fixed-length `jax.lax.scan` rollouts.

## 3. PRNG Management
PRNGKeys are passed explicitly to `env.reset(rng)`. During the step, stochasticity is minimal unless specifically designed into the environment.

## 4. Observation and Action Spaces
- **Spaces**: Environments define `observation_size` and `action_size`.
- **Typing**: Both are continuous arrays (JAX `jnp.float32`). Actions typically range between `[-1.0, 1.0]`.
