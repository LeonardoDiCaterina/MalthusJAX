"""Quality-Diversity tests for NEAT MAP-Elites engines."""

from __future__ import annotations

import numpy as np

from malthus_neat.learn.neat_classifier import NEATClassifier
from malthus_neat.learn.neat_regressor import NEATRegressor


def test_neat_map_elites_repertoire_filling() -> None:
    """Tests that NEAT MAP-Elites fills centroids in the 2D behavioral descriptor space."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    qd_reg = NEATRegressor(
        pop_size=10,
        max_generations=5,
        engine="map_elites",
        engine_params={"num_centroids": 20},
        random_state=42,
    )
    qd_reg.fit(X, y)

    assert hasattr(qd_reg, "repertoire_")
    rep = qd_reg.get_repertoire()
    assert rep.centroids.shape == (20, 2)
    assert hasattr(rep, "fitnesses")
    assert hasattr(rep, "descriptors")

    preds = qd_reg.predict(X)
    assert preds.shape == (4,)


def test_neat_island_map_elites() -> None:
    """Tests distributed Island MAP-Elites with NEAT networks."""
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([0.5, 0.5], dtype=np.float32)

    island_qd = NEATRegressor(
        pop_size=10,
        max_generations=6,
        engine="island_map_elites",
        engine_params={
            "num_islands": 2,
            "migration_interval": 3,
            "num_centroids": 15,
        },
        random_state=42,
    )
    island_qd.fit(X, y)
    assert hasattr(island_qd, "best_genome_")
    preds = island_qd.predict(X)
    assert preds.shape == (2,)


def test_neat_classifier_map_elites() -> None:
    """Tests MAP-Elites QD engine on classification problems."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    qd_clf = NEATClassifier(
        pop_size=10,
        max_generations=5,
        engine="map_elites",
        engine_params={"num_centroids": 10},
        random_state=42,
    )
    qd_clf.fit(X, y)
    assert hasattr(qd_clf, "repertoire_")
    probs = qd_clf.predict_proba(X)
    assert probs.shape == (4, 2)
