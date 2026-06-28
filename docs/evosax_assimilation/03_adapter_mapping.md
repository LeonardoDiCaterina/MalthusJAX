# Adapter Mapping (Evosax)

This document specifies the exact `@adapter` configuration required to assimilate Evosax's execution loop into MalthusJAX.

## The `@adapter` Definition

```python
from malthusjax.composer.adapters import adapter, EvalMode

@adapter(
    framework="evosax",
    state_mapping={
        "init": "init",
        "ask": "ask",
        "tell": "tell",
        # Evosax does not require a secondary `transform` step.
        "transform": None 
    },
    metrics_mapping={
        # Evosax natively tracks best_fitness in its state object
        "best_fitness": "best_fitness"
    },
    eval_translators={
        # Native Evosax evaluator requires state and rng_key
        EvalMode.NATIVE: lambda problem, state, pop_raw, keys, forward_fn: problem.evaluate(keys, pop_raw),
        
        # MalthusJAX evaluator simply takes the raw genotypes and evaluates them
        EvalMode.MALTHUSJAX: lambda mjx_eval, pop_raw: mjx_eval.evaluate_batch(pop_raw)
    }
)
class EvosaxEngineAdapter:
    """
    Universally adapted Evosax engine.
    """
    pass
```

### Key Differences from other Frameworks
- Evosax natively tracks metrics in its state via `EvoState.best_fitness`, meaning MalthusJAX can simply map `metrics_mapping={"best_fitness": "best_fitness"}`.
- The parameter structures (genotypes) returned by `ask` are natively executable arrays, bypassing the need for a `transform` method (unlike TensorNEAT).
