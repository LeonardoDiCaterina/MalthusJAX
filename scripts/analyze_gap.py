#!/usr/bin/env python
"""Analyze the performance gap between MalthusJAX and evosax."""

from malthusjax.operators.selection import ElitePoolSelection  
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxGaussianWrapper
from malthusjax.engine.resource_mapper import compute_resource_map
from malthusjax.core.genome.real_genome import RealGenomeConfig

config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
selection = ElitePoolSelection(elite_k=10, num_selections=500)
crossover = EvosaxUniformCrossoverWrapper()
mutation = EvosaxGaussianWrapper()

rmap = compute_resource_map(
    pop_size=500,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    genome_config=config,
)

print('=== Resource Map Analysis ===')
print(f'Total RNG budget: {rmap.total_rng_budget} keys')
print(f'  Selection: {rmap.selection.num_keys} keys')
print(f'  Crossover: {rmap.crossover.num_keys} keys')
print(f'  Mutation:  {rmap.mutation.num_keys} keys (injection_mode)')
print(f'  Next key:  {rmap.next_key.num_keys} key')

# Analyze the step overhead
print('\n=== Architecture Overhead Analysis ===')
print('Per-step operations in MalthusJAX GeneticEngine:')
print('  1. _allocate_entropy: jax.random.split(key, {}) keys'.format(rmap.total_rng_budget))
print('  2. _get_active_operators: update mutation strength (if scheduled)')
print('  3. _selection_phase: selection + elite extraction with tree_map')
print('  4. _reproduction_phase: crossover (nested vmap) + mutation (nested vmap)')
print('  5. _merge: dynamic_update_slice for elites + offspring')
print('  6. _evaluate: fitness evaluation')
print('  7. HOF update: track best individual')
print()
print('Per-step operations in evosax SimpleGA:')
print('  1. ask: sample new solutions (single vmap)')
print('  2. eval: evaluate fitness (vmap)')
print('  3. tell: update state')
print()
print('Key difference: MalthusJAX has 7 phases with many tree_maps vs evosax has 3 simple steps.')
