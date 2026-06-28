# Adapter Mapping (TensorNEAT)

This document specifies the exact `@adapter` configuration required to assimilate TensorNEAT's execution loop into MalthusJAX.

## The `@adapter` Definition

```python
from malthusjax.composer.adapters import adapter, EvalMode
import jax

@adapter(
    framework="tensorneat",
    state_mapping={
        "init": "setup",
        "ask": "ask",
        "tell": "tell",
        # TensorNEAT requires transform to build the neural network before execution
        "transform": "transform",
        "forward": "forward"
    },
    metrics_mapping={
        # Metrics must be manually calculated by the Universal Adapter since tell() doesn't return them
        "best_fitness": "max_fitness" 
    },
    eval_translators={
        # Native evaluator relies on jax.vmap mapping over the transformed network keys and populations
        EvalMode.NATIVE: lambda problem, state, pop_transformed, keys, forward_fn: (
            jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(state, keys, forward_fn, pop_transformed)
        ),
        
        # MalthusJAX evaluator evaluates the *transformed* network parameters
        EvalMode.MALTHUSJAX: lambda mjx_eval, pop_transformed: mjx_eval.evaluate_batch(pop_transformed)
    }
)
class TensorNEATEngineAdapter:
    """
    Universally adapted TensorNEAT engine.
    """
    pass
```

### Key Differences from other Frameworks
- **The `transform` property**: The adapter must intercept the genome `(nodes, conns)` output by `ask` and run the `algorithm.transform()` method before piping it to `evaluate`.
- **Manual Metrics**: Since `tell()` returns no metrics, the `@adapter` must compute standard metrics automatically by identifying `metrics_mapping={"best_fitness": "max_fitness"}`.
