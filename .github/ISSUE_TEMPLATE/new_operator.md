---
name: New Genetic Operator
about: Propose a new mutation, crossover, or selection operator
title: '[OPERATOR] '
labels: 'enhancement, operator'
assignees: ''
---

## Description
Describe the genetic operator you are proposing (e.g., a specific type of crossover, a domain-specific mutation). What problem does it solve?

## Mathematical / Algorithmic Foundation
Briefly explain the underlying algorithm. If this operator is based on a research paper, please link it here.

## Proposed API
How will this operator be configured? (e.g., `mutation_rate`, `tournament_size`)
```python
# Example
@register_mutation("my_mutation", compatible_genomes=["real"])
@struct.dataclass
class MyMutation(BaseMutation):
    mutation_rate: float = 0.1
    # ...
```

## JAX Compatibility Assessment
Does this operator require any dynamic control flow that might conflict with `jax.jit` or `jax.vmap`?

## Alternatives Considered
Are there existing operators in MalthusJAX that could achieve this with different parameters?
