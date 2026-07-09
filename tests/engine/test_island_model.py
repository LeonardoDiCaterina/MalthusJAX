import jax
import jax.numpy as jnp
import pytest
import chex

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome
from malthusjax.engine.base import AbstractEngine
from malthusjax.engine.island_model.topologies import RingTopologyIsland, FullyConnectedIsland
from flax import struct

@struct.dataclass
class DummyState:
    population: BasePopulation
    rng_key: chex.Array

@struct.dataclass
class DummyEngine(AbstractEngine):
    def init_state(self, key):
        # We need a fixed pop_size since the DummyEngine doesn't have engine_params configured cleanly in the test.
        # But wait, we can just hardcode pop_size=10 for the test
        pop_size = 10
        fitness = jnp.zeros(pop_size)
        genes = jnp.zeros((pop_size, 3))
        config = RealGenomeConfig(shape=(3,), bounds=(-1.0, 1.0))
        pop = BasePopulation(config=config, genes=RealGenome(values=genes), fitness=fitness)
        return DummyState(population=pop, rng_key=jax.random.split(key)[0])

    def step(self, state):
        # Dummy step just adds 1 to the genes
        genes = state.population.genes.values + 1.0
        new_pop = state.population.replace(genes=RealGenome(values=genes))
        return state.replace(population=new_pop), None

@pytest.fixture
def base_engine():
    return DummyEngine(engine_params=None)

def test_ring_topology_initialization(base_engine):
    island_model = RingTopologyIsland(
        engine=base_engine,
        num_islands=4,
        migration_interval=2,
        num_migrants=2
    )
    
    key = jax.random.PRNGKey(0)
    
    multi_state = island_model.init_state(key)
    multi_pop = multi_state.population
    
    # Check 2D structure
    assert isinstance(multi_pop, BasePopulation)
    assert multi_pop.fitness.shape == (4, 10)
    assert multi_pop.genes.values.shape == (4, 10, 3)
    
def test_ring_topology_migration(base_engine):
    island_model = RingTopologyIsland(
        engine=base_engine,
        num_islands=3,
        migration_interval=2,
        num_migrants=1
    )
    
    key = jax.random.PRNGKey(0)
    config = RealGenomeConfig(shape=(3,), bounds=(-1.0, 1.0))
    
    # We mock a population where islands have distinctly ranked fitnesses
    # Island 0: [0, 1, 2, 3]
    # Island 1: [10, 11, 12, 13]
    # Island 2: [20, 21, 22, 23]
    fitness = jnp.array([
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
    ])
    
    genes_values = jnp.zeros((3, 4, 3))
    # Best of island 0 is index 0 (val 0.0), worst is index 3 (val 3.0)
    # Ring shifts right. Island 1 should receive Island 0's best.
    
    genes = RealGenome(values=genes_values)
    multi_pop = BasePopulation(config=config, genes=genes, fitness=fitness)
    
    migrated_pop = island_model.migrate(key, multi_pop)
    
    # Best of Island 0 (fitness 0.0) should overwrite worst of Island 1 (fitness 13.0)
    # Island 1 original: [10.0, 11.0, 12.0, 13.0]. New: [10.0, 11.0, 12.0, 0.0]
    assert migrated_pop.fitness[1, 3] == 0.0
    
    # Best of Island 1 (fitness 10.0) should overwrite worst of Island 2 (fitness 23.0)
    assert migrated_pop.fitness[2, 3] == 10.0
    
    # Best of Island 2 (fitness 20.0) should overwrite worst of Island 0 (fitness 3.0)
    assert migrated_pop.fitness[0, 3] == 20.0

def test_fully_connected_migration(base_engine):
    island_model = FullyConnectedIsland(
        engine=base_engine,
        num_islands=4,
        migration_interval=2,
        num_migrants=2
    )
    
    key = jax.random.PRNGKey(0)
    config = RealGenomeConfig(shape=(3,), bounds=(-1.0, 1.0))
    
    fitness = jnp.arange(40).reshape(4, 10).astype(jnp.float32)
    genes_values = jnp.zeros((4, 10, 3))
    genes = RealGenome(values=genes_values)
    multi_pop = BasePopulation(config=config, genes=genes, fitness=fitness)
    
    migrated_pop = island_model.migrate(key, multi_pop)
    
    # The sum of all fitnesses should remain exactly the same (we just swapped elements)
    # Wait, no. We OVERWROTE the worst elements with copies of the best elements.
    # So the mean fitness should strictly decrease (improve).
    original_mean = jnp.mean(multi_pop.fitness)
    new_mean = jnp.mean(migrated_pop.fitness)
    
    assert new_mean < original_mean
    assert migrated_pop.fitness.shape == (4, 10)
