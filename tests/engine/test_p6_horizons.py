import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from malthusjax.engine.island_model.base import BaseIslandModel
from malthusjax.engine.island_model.adapter import IslandEngineAdapter
from tests.engine.test_island_model import DummyEngine, DummyEvaluator, DummyConfig

@struct.dataclass
class CountMigrations(BaseIslandModel):
    def migrate(self, key, pop):
        return pop.replace(fitness=pop.fitness + 1)

@pytest.mark.parametrize("generations", [0,2,4,10])
def test_exact_horizon_and_migrations(generations):
    engine = DummyEngine(engine_params=None, evaluator=DummyEvaluator(DummyConfig()))
    model = CountMigrations(engine, 2, 4, 1)
    adapter = IslandEngineAdapter(model, generations)
    initial = adapter.init_state(jax.random.key(42))
    state, _, _ = jax.jit(adapter.run)(initial)
    np.testing.assert_array_equal(state.population.genes.values, generations)
    np.testing.assert_array_equal(state.population.fitness, generations // 4)
    assert state.generation == generations

@pytest.mark.parametrize("kwargs", [{"num_islands":0}, {"migration_interval":0}, {"num_migrants":-1}])
def test_invalid_island_config(kwargs):
    config = dict(engine=None,num_islands=2,migration_interval=4,num_migrants=1) | kwargs
    with pytest.raises(ValueError):
        CountMigrations(**config)
