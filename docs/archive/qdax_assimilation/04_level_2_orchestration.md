# Phase 2: Level 2 Integration (Native Orchestration)

Level 2 integration drops the QDax `MAPElites.update` loop entirely. Instead, we plug the QDax **Emitter** and **Repertoire** directly into MalthusJAX's native `run()` engine loop.

This is critical for external/offline simulators where MalthusJAX must pause execution to wait for API responses.

## 1. The `QDEngine`

This engine manages the MalthusJAX `ask` and `tell` protocol but operates on `QDaxEngineState`.

```python
from malthusjax.engine.base import BaseEngine
import jax

class NativeQDEngine(BaseEngine):
    def __init__(self, evaluator, emitter, grid_shape):
        self.evaluator = evaluator # MalthusJAX Evaluator with .evaluate_qd()
        self.emitter = emitter     # Wrapped QDax Emitter Adapter
        self.grid_shape = grid_shape

    def ask(self, state: QDaxEngineState):
        """Asks the QDax Emitter for new genotypes."""
        key, subkey = jax.random.split(state.rng_key)
        genotypes, extra_info = self.emitter.emit(
            state.repertoire, 
            state.emitter_state, 
            subkey
        )
        new_state = state.replace(rng_key=key, extra_info=extra_info)
        return new_state, genotypes

    def tell(self, state: QDaxEngineState, genotypes, fitnesses, descriptors):
        """Adds to the Repertoire and updates the Emitter state."""
        
        new_repertoire = state.repertoire.add(
            genotypes, 
            descriptors, 
            fitnesses, 
            state.extra_info
        )
        
        new_emitter_state = self.emitter.state_update(
            state.emitter_state,
            new_repertoire,
            genotypes,
            fitnesses,
            descriptors,
            state.extra_info
        )
        
        return state.replace(
            repertoire=new_repertoire,
            emitter_state=new_emitter_state
        )
```

## 2. The Native Execution Loop

Because MalthusJAX owns the loop, we can compile it with `jax.lax.scan` for fast execution, *or* run it imperatively in Python for external evaluators.

```python
    # Inside NativeQDEngine.run()
    def _scan_fn(state, _):
        # 1. MalthusJAX Ask
        state, genotypes = self.ask(state)
        
        # 2. MalthusJAX Evaluate (Now supports descriptors)
        key, subk = jax.random.split(state.rng_key)
        fitnesses, descriptors = self.evaluator.evaluate_qd(genotypes, subk)
        state = state.replace(rng_key=key)
        
        # 3. MalthusJAX Tell
        state = self.tell(state, genotypes, fitnesses, descriptors)
        
        # 4. Tracking
        metrics = self.compute_metrics(state)
        return state, metrics
        
    final_state, history = jax.lax.scan(_scan_fn, state, (), length=num_generations)
```

By owning the `_scan_fn`, MalthusJAX can inject custom logging, hybrid operators, or callback mechanisms at any step of the generation, completely demystifying the QDax black box.
