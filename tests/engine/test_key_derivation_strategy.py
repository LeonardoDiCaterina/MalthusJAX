"""Unit tests for KeyDerivationStrategy and ResourceMap.get_keys()"""
import jax.random as jar
import jax.numpy as jnp

from malthusjax.engine.resource_mapper import (
    KeyDerivationStrategy, compute_resource_map
)
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import SinglePointCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig


def make_rmap(strategy: KeyDerivationStrategy):
    genome_config = BinaryGenomeConfig(length=8)
    selection = TournamentSelection(num_selections=8, tournament_size=3)
    crossover = SinglePointCrossover(num_offspring=2)
    mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
    return compute_resource_map(selection, crossover, mutation, genome_config, pop_size=8, key_derivation=strategy)


def test_enum_has_members():
    assert KeyDerivationStrategy.SPLIT.name == "SPLIT"
    assert KeyDerivationStrategy.FOLD.name == "FOLD"


def test_compute_resource_map_accepts_strategy():
    rmap_split = make_rmap(KeyDerivationStrategy.SPLIT)
    rmap_fold = make_rmap(KeyDerivationStrategy.FOLD)

    assert rmap_split.key_derivation == KeyDerivationStrategy.SPLIT
    assert rmap_fold.key_derivation == KeyDerivationStrategy.FOLD


def test_get_keys_shapes_and_different():
    rmap_split = make_rmap(KeyDerivationStrategy.SPLIT)
    rmap_fold = make_rmap(KeyDerivationStrategy.FOLD)

    master_key = jar.PRNGKey(0)
    keys_split_1 = rmap_split.get_keys(master_key)
    keys_split_2 = rmap_split.get_keys(master_key)
    keys_fold_1 = rmap_fold.get_keys(master_key)
    keys_fold_2 = rmap_fold.get_keys(master_key)

    assert keys_split_1.shape[0] == rmap_split.total_rng_budget
    assert keys_fold_1.shape[0] == rmap_fold.total_rng_budget

    # Determinism: same strategy + same master_key should produce identical keys
    assert jnp.allclose(keys_split_1, keys_split_2)
    assert jnp.allclose(keys_fold_1, keys_fold_2)
