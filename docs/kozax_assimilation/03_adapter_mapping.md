# Adapter Mapping (Kozax)

This document specifies the exact `@adapter` configuration required to assimilate Kozax's execution loop into MalthusJAX.

## The `@adapter` Definition

```python
from malthusjax.composer.adapters import adapter, EvalMode

@adapter(
    framework="kozax",
    state_mapping={
        "init": "initialize_population",
        # Monolithic frameworks map their step function to "step" instead of ask/tell
        "step": "evolve_population", 
        "transform": None
    },
    metrics_mapping={
        # Kozax expects Minimization. Metrics manually computed by Universal Adapter.
        "best_fitness": "min_fitness" 
    },
    eval_translators={
        # Kozax evaluate_population returns (fitness, optimized_population). 
        # We only want the fitness array (index 0).
        EvalMode.NATIVE: lambda gp, pop, data, key: gp.evaluate_population(pop, data, key)[0],
        
        # MalthusJAX evaluator simply takes the population tensor
        EvalMode.MALTHUSJAX: lambda mjx_eval, pop: mjx_eval.evaluate_batch(pop)
    }
)
class KozaxEngineAdapter:
    """
    Universally adapted Kozax engine.
    """
    pass
```

### Key Differences from other Frameworks
- **Stateless Operation**: Kozax does not return a `State` object from its init, but purely the population tensor. The `@adapter` loop will treat the tensor itself as the evolving "state" argument.
- **Monolithic `step` mapping**: Since Kozax uses `evolve_population` rather than split `ask` and `tell` methods, the adapter defines `"step": "evolve_population"`. The MalthusJAX engine loop knows to pass `(state, fitnesses, key)` to a `"step"` mapping, bypassing the standard ask/tell pipeline.
- **Minimization tracking**: Kozax seeks to minimize errors, so the metrics map expects `"min_fitness"`.
