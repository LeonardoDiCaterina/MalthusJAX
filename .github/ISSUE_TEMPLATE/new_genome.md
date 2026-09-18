---
name: New Genome Encoding
about: Propose a new genome representation or PyTree encoding
title: '[GENOME] '
labels: 'enhancement, core, genome'
assignees: ''
---

## Description
Describe the genome encoding you are proposing (e.g., Quaternion, Graph, Variable-length Linear, Permutation, Tree). What evolutionary domains or problems does it target?

## PyTree Data Structure & Schema
How is the genome structured as a JAX PyTree? (Note: all genome classes must inherit from `BaseGenome` and use `@struct.dataclass`).
- **Core Array(s)**: (e.g., `values: chex.Array`, `adjacency: chex.Array`)
- **Metadata Fields**: (static attributes marked `struct.field(pytree_node=False)`)

```python
# Proposed Genome Signature
@register_genome("my_genome")
@struct.dataclass
class MyGenome(BaseGenome[MyGenomeConfig]):
    values: chex.Array
    # ...
```

## Domain Constraints & Validation
- **Domain Bounds / Allowed Values**: 
- **Autocorrect / Projection Logic**: How are out-of-bounds or invalid genes corrected (e.g., clipping, normalization, modular wrap)?
- **Distance Metric**: What is the default distance function between two genomes (e.g., Euclidean, Hamming, structural distance)?

## Genetic Operator Compatibility
Which operator types are expected to operate on this genome?
- Selection: (All standard selections)
- Crossover: 
- Mutation: 

## JAX Compilation & Shape Contracts
Does this representation require fixed-shape padding to support `jax.jit` and `jax.vmap`?
- [ ] Yes (uses max-size padding and masking)
- [ ] No (natively fixed dimension)
