# Gymnax Framework Mechanics

This document breaks down the internal mechanics of the Gymnax framework and how its state and execution loop operate.

## 1. Initialization Protocol
- **Primary Object**: The environment is instantiated via a factory function: `env, env_params = gymnax.make(env_name)`
- **Reset Method**: `obs, state = env.reset(rng_key, env_params)`
- **State Object**: Gymnax uses environment-specific `EnvState` structs (e.g., `CartPoleState`). This state tracks internal variables like physics components or episodic step counts.

## 2. Execution Loop (Step Protocol)
Gymnax environments are purely functional.
- **Step Signature**: `obs, state, reward, done, info = env.step(rng_key, state, action, env_params)`
  - `rng_key`: Required for stochastic environments or action noise.
  - `state`: The `EnvState` from the previous step or reset.
  - `action`: The control input (discrete integer or continuous array).
- **Auto-Reset**: Gymnax environments do not automatically reset upon termination by default. This must be handled explicitly if required, or wrapped in an `AutoResetWrapper`.

## 3. PRNG Management
PRNGKeys are explicitly passed as arguments to both `reset` and `step` methods. JAX `rng_key` must be split before every step to ensure proper stochasticity.

## 4. Observation and Action Spaces
- **Spaces**: Environments provide `env.observation_space(env_params)` and `env.action_space(env_params)`.
- **Typing**: Observations and actions are purely JAX arrays (no dictionaries unless explicitly wrapped).
