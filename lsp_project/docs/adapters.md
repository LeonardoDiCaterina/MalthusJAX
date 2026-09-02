# Engine Adapters

MalthusJAX 2.0 provides an agnostic `GeneticEngine` that operates over any PyTree conforming to the correct structural contracts. To simplify the instantiation of complex evolutionary pipelines, this project utilizes specialized **Adapters**.

## MalthusJAX Engine Integration

The Adapters (e.g., `DMEPAdapter`, `CGPAdapter`, `LSMFAdapter`) act as high-level factories. They abstract away the boilerplate of instantiating genomes, evaluators, and operators, directly yielding a fully configured `GeneticEngine`.

### Example Pipeline

Below is a conceptual snippet of how an adapter configures the pipeline:

```python
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.schedules import TrackBest, TrackMetrics

# 1. Adapter configures the architectures and evaluators internally
gc = NeuralPrefixGenomeConfig(...)
ec = NeuralPrefixEvaluatorConfig(...)
lsmf_eval = LSMFEvaluator(...)

# 2. Engine instantiation
engine_params = GeneticEngineParams(
    pop_size=128, 
    num_generations=100,
    track_best=TrackBest.FULL,
    track_metrics=TrackMetrics.ALL
)

engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=gc,
    evaluator=lsmf_eval,
    mutation=HybridMutation(...),
    ...
)

# 3. Fully Compiled XLA Execution
final_state, history, metrics = jax.jit(engine.run)(initial_state)
```

By decoupling the pipeline assembly from the core execution, the adapters make it easy to swap between dMEP and dCGPANN simply by calling a different factory function.
