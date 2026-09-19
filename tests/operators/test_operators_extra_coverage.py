"""Targeted coverage tests for operators/selection/evosax_mimic.py and operators/mutation/categorical.py."""

import jax
import jax.numpy as jnp

from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.operators.mutation.categorical import ScrambleMutation, SwapMutation
from malthusjax.operators.selection.evosax_mimic import EvoSaxMimicSelection


def test_evosax_mimic_selection():
    sel = EvoSaxMimicSelection(num_selections=4, elite_k=3, n_elites=2)
    assert sel.num_keys_per_atomic_operation == 1

    key = jax.random.PRNGKey(0)
    fitness = jnp.array([10.0, 2.0, 5.0, 1.0, 8.0, 3.0])

    # Call with array
    parents, elites = sel(key, fitness)
    assert parents.shape == (4,)
    assert elites.shape == (2,)

    # Call with population-like object having .fitness attribute
    class DummyPop:
        def __init__(self, fit):
            self.fitness = fit

    pop = DummyPop(fitness)
    p2, e2 = sel(key, pop)
    assert p2.shape == (4,)
    assert e2.shape == (2,)


def test_categorical_scramble_mutation():
    mut = ScrambleMutation(mutation_rate=1.0)
    assert mut.num_keys_per_atomic_operation == 2

    config = CategoricalGenomeConfig(num_categories=10, shape=(6,))
    genome = CategoricalGenome(values=jnp.arange(6))

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)
    noise = mut._generate_noise(keys, config)
    mutated = mut._mutate_one(genome, noise, config)

    assert mutated.values.shape == (6,)
    # Permutation preserves set of values
    assert set(mutated.values.tolist()) == set(genome.values.tolist())


def test_categorical_swap_mutation():
    mut = SwapMutation(mutation_rate=1.0)
    assert mut.num_keys_per_atomic_operation == 3

    config = CategoricalGenomeConfig(num_categories=10, shape=(6,))
    genome = CategoricalGenome(values=jnp.arange(6))

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    noise = mut._generate_noise(keys, config)
    mutated = mut._mutate_one(genome, noise, config)

    assert mutated.values.shape == (6,)
    assert set(mutated.values.tolist()) == set(genome.values.tolist())
