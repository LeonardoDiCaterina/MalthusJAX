import jax
import numpy as np
import pytest
from flax import struct

from malthusjax.engine.island_model.adapter import IslandEngineAdapter
from malthusjax.engine.island_model.base import BaseIslandModel
from tests.engine.test_island_model import DummyConfig, DummyEngine, DummyEvaluator


@struct.dataclass
class CountMigrations(BaseIslandModel):
    def migrate(self, key, pop):
        return pop.replace(fitness=pop.fitness + 1)


@pytest.mark.parametrize("generations", [0, 2, 4, 10])
def test_exact_horizon_and_migrations(generations):
    engine = DummyEngine(engine_params=None, evaluator=DummyEvaluator(DummyConfig()))
    model = CountMigrations(engine, 2, 4, 1)
    adapter = IslandEngineAdapter(model, generations)
    initial = adapter.init_state(jax.random.key(42))
    state, _, _ = jax.jit(adapter.run)(initial)
    np.testing.assert_array_equal(state.population.genes.values, generations)
    np.testing.assert_array_equal(state.population.fitness, generations // 4)
    assert state.generation == generations


@pytest.mark.parametrize(
    "kwargs", [{"num_islands": 0}, {"migration_interval": 0}, {"num_migrants": -1}]
)
def test_invalid_island_config(kwargs):
    config = dict(engine=None, num_islands=2, migration_interval=4, num_migrants=1) | kwargs
    with pytest.raises(ValueError):
        CountMigrations(**config)


def test_island_output_modes_preserve_remainder_and_winner():
    from malthusjax.composer import EngineFactory, EngineOutputOptions, ExperimentConfig
    from malthusjax.engine.island_model.topologies import RingTopologyIsland

    config = ExperimentConfig.from_quick_run_kwargs(
        fitness="native_sphere",
        genome_length=3,
        pop_size=8,
        generations=5,
    )
    engine = EngineFactory.build(config).genetic_engine
    adapter = IslandEngineAdapter(RingTopologyIsland(engine, 2, 4, 1), generations=5)
    key = jax.random.key(4)
    full = adapter.run_with_options(key, EngineOutputOptions())
    reduced = adapter.run_with_options(
        key, EngineOutputOptions(collect_history=False, capture_solution=False)
    )
    assert full["summary"] == reduced["summary"]
    assert full["solution"]["fitness"] == full["summary"]["best_fitness"]
    assert len(full["history"]) == 5
    assert reduced["history"] == []
    assert "solution" not in reduced
    initial = adapter.init_state(key)
    _, history, _ = jax.jit(lambda state: adapter.run(state, return_history=False))(initial)
    assert history is None
    np.testing.assert_array_equal(initial.generation, [0, 0])
