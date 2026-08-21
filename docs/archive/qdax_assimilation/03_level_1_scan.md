# Phase 1: Level 1 Integration (The Scan Adapter)

The Level 1 adapter completely delegates the `jax.lax.scan` execution loop to QDax. MalthusJAX only handles the orchestration, setup, and teardown.

## 1. The `QDaxEngineAdapter`

This adapter implements MalthusJAX's `BaseEngine` but initializes and manages a `qdax.MAPElites` instance internally.

```python
from malthusjax.engine.base import BaseEngine
from qdax.core.map_elites import MAPElites
import jax

class QDaxEngineAdapter(BaseEngine):
    def __init__(self, scoring_function, emitter, metrics_function):
        self.map_elites = MAPElites(
            scoring_function=scoring_function,
            emitter=emitter,
            metrics_function=metrics_function
        )
        
    def init_state(self, key, initial_genotypes, centroids):
        repertoire, emitter_state, _ = self.map_elites.init(
            initial_genotypes, 
            centroids, 
            key
        )
        return QDaxEngineState(repertoire=repertoire, emitter_state=emitter_state)

    def run(self, state: QDaxEngineState, key, num_generations: int):
        # Delegate the entire loop to QDax's XLA compiled scan
        def _scan_fn(carry, _):
            rep, em_state, k = carry
            k, subk = jax.random.split(k)
            rep, em_state, metrics = self.map_elites.update(rep, em_state, subk)
            return (rep, em_state, k), metrics
            
        initial_carry = (state.repertoire, state.emitter_state, key)
        final_carry, history = jax.lax.scan(_scan_fn, initial_carry, (), length=num_generations)
        
        final_rep, final_em_state, final_key = final_carry
        
        return QDaxEngineState(repertoire=final_rep, emitter_state=final_em_state), history
```

## 2. Integration with Composer

By wrapping QDax components in MalthusJAX `@register_` decorators, we enable users to configure the adapter via TOML.

```python
from malthusjax.composer.registry import register_engine

@register_engine("qdax_map_elites_adapter")
def create_qdax_adapter(evaluator_name, emitter_name, **kwargs):
    # Retrieve instances from MalthusJAX registry
    evaluator = registry.get_evaluator(evaluator_name)
    emitter = registry.get_emitter(emitter_name)
    
    return QDaxEngineAdapter(
        scoring_function=evaluator.scoring_fn, # Raw QDax scoring function
        emitter=emitter.qdax_emitter,          # Raw QDax emitter
        metrics_function=default_qd_metrics
    )
```

**Result**: A functional QDax MAP-Elites experiment triggerable via `mjax run configs/qdax_experiment.toml`.
