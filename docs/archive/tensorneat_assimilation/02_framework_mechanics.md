# TensorNEAT Framework Mechanics

This document breaks down the internal mechanics of the TensorNEAT framework.

## 1. Initialization Protocol
- **Primary Object**: `tensorneat.algorithm.BaseAlgorithm` (e.g. `NEAT`)
- **Init Method**: `state = algorithm.setup(state)` where `state` is a pre-initialized `tensorneat.common.State()` wrapper.
- **State Object**: TensorNEAT uses a `State` object which acts like an immutable dictionary to wrap all PRNGKeys and configuration parameters. The initial seed must be registered to the state before calling `setup`.

## 2. Execution Loop (Step Protocol)
TensorNEAT introduces a mandatory "transform" phase between `ask` and `eval`.
- **Architecture Type**: Ask -> Transform -> Eval -> Tell
- **Ask Signature**: `pop = algorithm.ask(state)`
  - Returns a tuple of arrays `(nodes, conns)` representing the raw topology encoding.
- **Transform Signature**: `transformed_pop = algorithm.transform(state, pop)`
  - Converts the raw topology encoding into usable neural network parameters/masks capable of processing inputs.
- **Tell Signature**: `state = algorithm.tell(state, fitnesses)`
  - Updates the NEAT population species, crossover statistics, and node/connection innovations.

## 3. PRNG Management
PRNG is strictly managed within the `tensorneat.common.State` wrapper via `state.randkey`. External loops interacting with TensorNEAT must extract, split, and re-inject the key into the state manually: `state = state.update(randkey=new_key)`.

## 4. Metrics & Logging
- **Missing Metrics**: Unlike Evosax, `algorithm.tell()` does **not** return a metrics dictionary. It just returns the updated state.
- TensorNEAT expects external scripts (like its native `Pipeline` class) to manually calculate `min`, `max`, `mean`, and `std` from the fitness array at every generation.
- **Maximization**: TensorNEAT natively assumes **Maximization** (it tracks best fitness up, penalizing failed genomes with `-inf`).
