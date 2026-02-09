"""Tests for ResourceMap.get_keys() across PRNG implementations and derivation strategies."""

import jax
import jax.random as jr
import pytest

from malthusjax.core.random import create_key, PRNGImpl
from malthusjax.engine.resource_mapper import (
    KeyDerivationStrategy,
    compute_resource_map,
)
from malthusjax.operators.crossover.binary import SinglePointCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig


def make_rmap(strategy: KeyDerivationStrategy):
    genome_config = BinaryGenomeConfig(shape=(8,))
    selection = TournamentSelection(num_selections=8, tournament_size=3)
    crossover = SinglePointCrossover(num_offspring=2)
    mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
    return compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size=8, key_derivation=strategy
    )


@pytest.mark.parametrize("impl", list(PRNGImpl))
def test_get_keys_shapes_and_different(impl):
    try:
        master = create_key(0, impl=impl)
    except ValueError:
        pytest.skip(f"impl {impl} not supported by this JAX build")

    rmap_split = make_rmap(KeyDerivationStrategy.SPLIT)
    rmap_fold = make_rmap(KeyDerivationStrategy.FOLD)

    keys_split_1 = rmap_split.get_keys(master)
    keys_split_2 = rmap_split.get_keys(master)
    keys_fold_1 = rmap_fold.get_keys(master)
    keys_fold_2 = rmap_fold.get_keys(master)

    assert keys_split_1.shape[0] == rmap_split.total_rng_budget
    assert keys_fold_1.shape[0] == rmap_fold.total_rng_budget

    # Determinism: same strategy + same master_key should produce identical keys
    assert jax.numpy.allclose(keys_split_1, keys_split_2)
    assert jax.numpy.allclose(keys_fold_1, keys_fold_2)


@pytest.mark.parametrize("impl", list(PRNGImpl))
def test_key_slices_are_distinct(impl):
    try:
        master = create_key(1, impl=impl)
    except ValueError:
        pytest.skip(f"impl {impl} not supported by this JAX build")

    rmap = make_rmap(KeyDerivationStrategy.SPLIT)
    all_keys = rmap.get_keys(master)

    sel_slice = rmap.get_key_slice("selection")
    cross_slice = rmap.get_key_slice("crossover")
    mut_slice = rmap.get_key_slice("mutation")
    nxt_slice = rmap.get_key_slice("next_key")

    # Ensure slices do not overlap
    slices = [sel_slice, cross_slice, mut_slice, nxt_slice]
    indices = set()
    for s in slices:
        for i in range(s.start, s.stop):
            assert i not in indices
            indices.add(i)


@pytest.mark.parametrize("strategy", [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD])
@pytest.mark.parametrize("impl", list(PRNGImpl))
def test_all_impls_x_strategies(impl, strategy):
    try:
        master = create_key(7, impl=impl)
    except ValueError:
        pytest.skip(f"impl {impl} not supported by this JAX build")

    rmap = make_rmap(strategy)
    keys = rmap.get_keys(master)
    assert keys.shape[0] == rmap.total_rng_budget
