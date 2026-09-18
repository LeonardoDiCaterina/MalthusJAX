---
name: New Framework Adapter
about: Propose a new third-party library or framework adapter for the MalthusJAX Composer
title: '[ADAPTER] '
labels: 'enhancement, composer, adapter'
assignees: ''
---

## Description
Describe the external framework or evolutionary library you are proposing to adapt (e.g., EvoSAX, QDAX, TensorNEAT, Kozax, PyPop7). What unique algorithms, representations, or benchmark problems does it bring to MalthusJAX?

## Framework Details
- **Framework Name**: 
- **Repository / Documentation URL**: 
- **JAX Native**: [ ] Yes / [ ] No (requires CPU fallback / host callback)
- **Execution Architecture**: [ ] Ask / Tell / [ ] Step / [ ] JIT Scan Loop

## State & Population Mapping
How does this framework represent its search population and algorithm state?
- **Genome / Representation**: (e.g., flat `jax.Array`, custom PyTree, graph structure)
- **Population Conversion**: How will individuals be converted to/from MalthusJAX genomes (`RealGenome`, `BinaryGenome`, etc.) for fair comparisons?

```python
# Proposed Adapter Signature or Structure
@adapter(framework="my_framework", strategy_cls=MyFrameworkStrategy)
def build_my_framework_engine(config, evaluator, **kwargs):
    # ...
```

## Evaluation Translators (`EvalMode`)
Which evaluation modes should be supported?
- [ ] **`EvalMode.MJX`**: Run MalthusJAX composable evaluators on raw framework population tensors.
- [ ] **`EvalMode.NATIVE`**: Run the framework's native objective / problem function and translate metrics into MalthusJAX formats.

## Metric Specs & Logging
What metrics will be logged per generation?
- Standard: `best_fitness`, `mean_fitness`, `generation_time`
- Specialized: (e.g., QD archive coverage, novelty scores, Pareto hypervolume)

## Alternatives Considered
Can this algorithm or workflow already be achieved using native MalthusJAX engines (`GeneticEngine`, `MOEngine`, `MapElitesEngine`)?
