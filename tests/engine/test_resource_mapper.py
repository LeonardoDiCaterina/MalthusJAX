import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from malthusjax.engine.resource_mapper import (
    ResourceMap, ShardingManager, KeyDerivationStrategy, compute_resource_map,
    get_step_dimension_flow, get_resource_summary
)
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.base import BaseCrossover
from malthusjax.operators.mutation.real import GaussianMutation

def test_sharding_manager_alloc_and_split():
    manager = ShardingManager()
    pop = manager.alloc_population((4, 2))
    assert pop.shape == (4, 2)
    
    key = jax.random.PRNGKey(0)
    keys = manager.split_key_sharded(key, 4)
    assert keys.shape == (4, 2)

def test_get_output_count():
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1
    )
    assert rmap.get_output_count("selection") == 6

def test_unknown_key_derivation_strategy():
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1
    )
    rmap = rmap.replace(key_derivation="UNKNOWN")
    with pytest.raises(ValueError, match="Unknown key derivation strategy"):
        rmap.get_keys(jax.random.PRNGKey(0))

def test_fold_in_incompatible_backend():
    old_val = getattr(jax.config, "jax_default_prng_impl", "threefry2x32")
    try:
        jax.config.update("jax_default_prng_impl", "rbg")
        with pytest.raises(ValueError, match="incompatible with PRNG impl"):
            compute_resource_map(
                selection=TournamentSelection(num_selections=4, tournament_size=2),
                crossover=UniformCrossover(),
                mutation=GaussianMutation(),
                genome_config=RealGenomeConfig(shape=(2,)),
                pop_size=4,
                key_derivation=KeyDerivationStrategy.FOLD
            )
    finally:
        jax.config.update("jax_default_prng_impl", old_val)

def test_genome_config_length():
    class DummyConfigLength:
        length = 5
        
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=DummyConfigLength(),
        pop_size=4
    )
    assert rmap.genome_shape == (5,)

def test_genome_config_size():
    class DummyConfigSize:
        size = 5
        
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=DummyConfigSize(),
        pop_size=4
    )
    assert rmap.genome_shape == (5,)

def test_get_step_dimension_flow():
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1
    )
    flow = get_step_dimension_flow(rmap, elitism=1, genome_width=2)
    assert "Step Dimension Flow:" in flow
    assert "p = ceil" in flow

def test_get_key_slice():
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1
    )
    s = rmap.get_key_slice("selection")
    assert isinstance(s, slice)

def test_get_keys_split_and_fold():
    rmap_split = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1,
        key_derivation=KeyDerivationStrategy.SPLIT
    )
    master_key = jax.random.PRNGKey(0)
    keys_split = rmap_split.get_keys(master_key)
    assert keys_split.shape == (rmap_split.total_rng_budget, 2)
    
    rmap_fold = rmap_split.replace(key_derivation=KeyDerivationStrategy.FOLD)
    keys_fold = rmap_fold.get_keys(master_key)
    assert keys_fold.shape == (rmap_split.total_rng_budget, 2)

def test_genome_config_resolved_shape():
    class DummyConfigResolved:
        resolved_shape = (7, 7)
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=DummyConfigResolved(),
        pop_size=4
    )
    assert rmap.genome_shape == (7, 7)

def test_genome_config_empty_shape():
    class DummyConfigEmpty:
        pass
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=DummyConfigEmpty(),
        pop_size=4
    )
    assert rmap.genome_shape == ()

def test_overproduction_warning(caplog):
    import logging
    from flax import struct

    @struct.dataclass
    class HighOffspringCrossover(BaseCrossover):
        num_offspring: int = 4
        
        def num_keys(self, input_shape=None): return 1
        
        def apply(self, genes, key): return genes
        def _get_batch_axes(self): return None

    with caplog.at_level(logging.WARNING):
        rmap = compute_resource_map(
            selection=TournamentSelection(num_selections=4, tournament_size=2),
            crossover=HighOffspringCrossover(),
            mutation=GaussianMutation(),
            genome_config=RealGenomeConfig(shape=(2,)),
            pop_size=5,
            elitism=0
        )
    assert rmap.crossover.output_count > 5
    assert any("overproduction ratio" in record.message for record in caplog.records)

def test_get_resource_summary():
    rmap = compute_resource_map(
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        genome_config=RealGenomeConfig(shape=(2,)),
        pop_size=4,
        elitism=1
    )
    summary = get_resource_summary(rmap)
    assert "Pipeline Resource & Flow Summary" in summary

