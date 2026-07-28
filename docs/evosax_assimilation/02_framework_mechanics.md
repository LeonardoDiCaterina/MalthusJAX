# Evosax Framework Mechanics

This document breaks down the internal mechanics of the Evosax framework and how its state and execution loop operate.

## 1. Initialization Protocol
- **Primary Object**: `evosax.Strategy` (e.g., `CMA_ES`, `OpenES`)
- **Init Method**: `state = strategy.init(rng_key, strategy_params)`
- **State Object**: Evosax uses `EvoState`, a flax `struct.dataclass`. This state object contains all internal variables for the evolutionary strategy (e.g., means, covariances, best fitness trackers).

## 2. Execution Loop (Step Protocol)
Evosax separates the generation process into two distinct phases (Ask and Tell).
- **Architecture Type**: Split (`ask` -> `eval` -> `tell`)
- **Ask Signature**: `x, state = strategy.ask(rng_key, state, strategy_params)`
  - Returns raw parameter genotypes (`x`) of shape `(pop_size, num_dims)`.
- **Tell Signature**: `state = strategy.tell(x, fitness, state, strategy_params)`
  - Accepts the evaluated fitness array and updates the internal parameters (e.g., covariance matrices) of the `EvoState`.

## 3. PRNG Management
PRNGKeys are explicitly passed as arguments to `init` and `ask` methods. Evosax does not embed or mutate keys directly inside the `EvoState` struct; it expects the caller to manage `jax.random.split`.

## 4. Metrics & Logging
- **Metrics**: The `EvoState` natively tracks the best member (`state.best_member`) and best fitness (`state.best_fitness`) over time.
- Evosax assumes **Minimization** or **Maximization** depending on the specific implementation, but conventionally, evolutionary algorithms treat lower as better if evaluating loss, or higher as better for rewards. This is managed by MalthusJAX's universal engine tracking.
