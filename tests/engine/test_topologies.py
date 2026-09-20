import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.engine.island_model.topologies import (
    FullyConnectedIsland,
    RingTopologyIsland,
)


class MockEngine:
    maximize = True


@pytest.fixture
def multi_pop():
    genes = jnp.array(
        [
            [[1.0], [2.0]],  # island 0
            [[3.0], [4.0]],  # island 1
            [[5.0], [6.0]],  # island 2
        ]
    )
    fitness = jnp.array(
        [
            [0.1, 0.9],  # island 0
            [0.3, 0.7],  # island 1
            [0.5, 0.5],  # island 2
        ]
    )
    return BasePopulation(genes=genes, fitness=fitness)


def test_ring_topology(multi_pop):
    topology = RingTopologyIsland(
        engine=MockEngine(), num_islands=3, migration_interval=1, num_migrants=1
    )
    key = jax.random.PRNGKey(0)
    migrated_pop = topology.migrate(key, multi_pop)
    assert migrated_pop.genes.shape == (3, 2, 1)


def test_fully_connected_topology(multi_pop):
    topology = FullyConnectedIsland(
        engine=MockEngine(), num_islands=3, migration_interval=1, num_migrants=1
    )
    key = jax.random.PRNGKey(0)
    migrated_pop = topology.migrate(key, multi_pop)
    assert migrated_pop.genes.shape == (3, 2, 1)
