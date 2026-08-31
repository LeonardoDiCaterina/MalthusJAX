import jax.numpy as jnp
import pytest

from malthusjax.composer.mo_factory import MOEngineAdapter, build_mo_engine
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig


class DummyEvaluatorConfig:
    maximize: bool = False


class DummyEvaluator:
    def __init__(self, maximize=False):
        self.config = DummyEvaluatorConfig()
        self.config.maximize = maximize


class DummyEmitter:
    pass


def test_build_mo_engine_real_genome():
    evaluator = DummyEvaluator()
    emitter = DummyEmitter()

    adapter = build_mo_engine(
        fitness_evaluator=evaluator,
        emitter=emitter,
        genome_type="real",
        pop_size=12,
        generations=3,
        genome_length=5,
        bounds=(-1.0, 1.0),
    )

    assert isinstance(adapter, MOEngineAdapter)
    assert adapter.maximize is False
    assert isinstance(adapter.genome_config, RealGenomeConfig)
    assert adapter.genome_config.shape == (5,)
    assert adapter.genome_config.bounds == (-1.0, 1.0)


def test_build_mo_engine_binary_genome():
    evaluator = DummyEvaluator(maximize=True)
    emitter = DummyEmitter()

    adapter = build_mo_engine(
        fitness_evaluator=evaluator,
        emitter=emitter,
        genome_type="binary",
        pop_size=10,
        generations=2,
        genome_length=8,
    )

    assert isinstance(adapter, MOEngineAdapter)
    assert adapter.maximize is True
    assert isinstance(adapter.genome_config, BinaryGenomeConfig)
    assert adapter.genome_config.shape == (8,)


def test_build_mo_engine_unsupported_genome():
    evaluator = DummyEvaluator()
    emitter = DummyEmitter()

    with pytest.raises(ValueError, match="Unsupported genome type: unknown"):
        build_mo_engine(
            fitness_evaluator=evaluator,
            emitter=emitter,
            genome_type="unknown",
            pop_size=10,
            generations=2,
        )


def test_build_mo_engine_initial_population():
    evaluator = DummyEvaluator()
    emitter = DummyEmitter()
    init_pop = jnp.zeros((10, 3))

    adapter = build_mo_engine(
        fitness_evaluator=evaluator,
        emitter=emitter,
        genome_type="real",
        pop_size=10,
        generations=2,
        genome_shape=(3,),
        initial_population=init_pop,
    )

    assert adapter.initial_population is not None
    assert jnp.array_equal(adapter.initial_population, init_pop)
