---
name: New Evolutionary Engine
about: Propose a new evolutionary engine or strategy (e.g., from evosax)
title: '[ENGINE] '
labels: 'enhancement, engine'
assignees: ''
---

## Description
Describe the evolutionary engine you are proposing (e.g., CMA-ES, MAP-Elites, SNES). 

## Evosax Parity
Is this engine aiming to match an existing strategy in the `evosax` library? 
- [ ] Yes (Please link the evosax documentation or source)
- [ ] No

## Mathematical / Algorithmic Foundation
Briefly explain the underlying algorithm and update rules. If based on a paper, please link it.

## Proposed State & Config
What will the engine's State and Configuration look like? (Keep in mind everything must be a `flax.struct.dataclass`).

```python
# Example
@struct.dataclass
class MyEngineState:
    mean: jax.Array
    covariance: jax.Array
    # ...
```

## Operator Compatibility
Does this engine rely on standard MalthusJAX operators (Selection, Mutation, Crossover) or does it handle population updates natively within its `step()` function?
