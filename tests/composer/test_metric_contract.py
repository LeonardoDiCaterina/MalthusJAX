"""Reported statistics match the actual archive, including maximization."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.composer import Composer
from malthusjax.composer.engine_factory import build_engine
from malthusjax.composer.engine_factory_v2 import EngineFactory
from malthusjax.composer.experiment_config import ExperimentConfig
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter


@pytest.mark.parametrize("maximize", [False, True])
def test_native_reporting_matches_actual_occupied_archive(maximize):
    pytest.importorskip("qdax")
    Composer.create_default()
    cfg = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0))
    emitter = GeneticMutationEmitter(
        _batch_size=4,
        mutation=OperatorCatalog().get("gaussian:mutation_rate=1.0,mutation_strength=0.1"),
        genome_config=cfg,
    )
    config = ExperimentConfig.from_quick_run_kwargs(
        fitness="sphere:dim=2",
        pop_size=4,
        generations=2,
        genome_length=2,
        maximize=maximize,
        bounds=(-5.0, 5.0),
        strategy=MapElitesStrategy(
            emitter=emitter,
            num_descriptors=2,
            num_centroids=3,
            centroids=jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ),
    )
    adapter = EngineFactory.build(config)
    captured = []
    engine_type = type(adapter.engine)
    original = engine_type.run

    def observe(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(result[0].repertoire.fitnesses)
        return result

    with patch.object(engine_type, "run", observe):
        result = adapter.run_once(jax.random.PRNGKey(0))
    archive = np.asarray(captured[0])
    occupied = archive[np.isfinite(archive)]
    assert occupied.size > 0
    sign = 1.0 if maximize else -1.0
    assert result["summary"]["best_fitness"] == pytest.approx(sign * np.max(occupied))
    assert result["summary"]["mean_fitness"] == pytest.approx(sign * np.mean(occupied))
    assert result["summary"]["std_fitness"] == pytest.approx(np.std(occupied), abs=1e-5)
    assert result["summary"]["qd_score"] == pytest.approx(np.sum(occupied))


def test_genetic_maximization_standard_deviations_are_nonnegative():
    catalog = OperatorCatalog()
    adapter = build_engine(
        fitness_evaluator=catalog.get("sphere:dim=2", maximize=True),
        selection_op=catalog.get("tournament:num_selections=4,tournament_size=2"),
        crossover_op=catalog.get("blend"),
        mutation_op=catalog.get("gaussian:mutation_rate=0.2"),
        pop_size=4,
        generations=2,
        genome_length=2,
        maximize=True,
    )
    result = adapter.run_once(jax.random.PRNGKey(0))
    assert all(row["std_fitness"] >= 0 for row in result["history"])
