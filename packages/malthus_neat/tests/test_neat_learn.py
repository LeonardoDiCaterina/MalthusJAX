"""Unit and integration tests for NEAT Scikit-Learn estimators."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from malthus_neat.learn.base import BaseNEATEstimator
from malthus_neat.learn.neat_classifier import NEATClassifier
from malthus_neat.learn.neat_regressor import NEATRegressor


def test_neat_regressor_synthetic() -> None:
    """Tests NEATRegressor fitting, prediction, and R^2 scoring on non-linear data."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1).astype(np.float32)
    y = (X[:, 0] ** 2).astype(np.float32)

    reg = NEATRegressor(
        pop_size=10,
        max_generations=5,
        max_nodes=15,
        max_conns=30,
        random_state=42,
    )
    reg.fit(X, y)

    assert hasattr(reg, "best_genome_")
    assert isinstance(reg.best_fitness_, float)

    preds = reg.predict(X)
    assert preds.shape == (20,)
    assert not np.isnan(preds).any()

    r2 = reg.score(X, y)
    assert isinstance(r2, float)


def test_neat_classifier_xor() -> None:
    """Tests NEATClassifier fitting, predict_proba, and classification score on XOR."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    clf = NEATClassifier(
        pop_size=10,
        max_generations=5,
        random_state=42,
    )
    clf.fit(X, y)

    assert hasattr(clf, "best_genome_")
    probs = clf.predict_proba(X)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)

    preds = clf.predict(X)
    assert preds.shape == (4,)

    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_neat_toml_serialization_roundtrip() -> None:
    """Tests full roundtrip serialization and state restoration via TOML."""
    X = np.array([[0.5, -0.2], [-0.1, 0.8], [0.3, 0.3]], dtype=np.float32)
    y = np.array([0.3, 0.7, 0.6], dtype=np.float32)

    reg = NEATRegressor(pop_size=8, max_generations=4, random_state=123)
    reg.fit(X, y)
    original_preds = reg.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = Path(tmpdir) / "test_neat_model.toml"
        toml_content = reg.to_toml(filepath=toml_path)
        assert "NEATRegressor" in toml_content

        # Load from file
        loaded_from_file = BaseNEATEstimator.from_toml(toml_path)
        preds_file = loaded_from_file.predict(X)
        np.testing.assert_allclose(original_preds, preds_file, rtol=1e-5)

        # Load from string
        loaded_from_str = BaseNEATEstimator.from_toml(toml_content)
        preds_str = loaded_from_str.predict(X)
        np.testing.assert_allclose(original_preds, preds_str, rtol=1e-5)


def test_neat_visualization_and_graph_exports() -> None:
    """Tests DOT generation, NetworkX conversion, and file visualization."""
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0.5, 0.5], dtype=np.float32)

    reg = NEATRegressor(pop_size=6, max_generations=2, random_state=7)
    reg.fit(X, y)

    dot_str = reg.to_dot()
    assert "digraph NEAT" in dot_str
    assert "subgraph" in dot_str

    G = reg.to_networkx()
    assert len(G.nodes) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = Path(tmpdir) / "topology.svg"
        reg.visualize(filepath=svg_path, format="svg")
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0


def test_neat_island_engine() -> None:
    """Tests island model execution with NEAT."""
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y = np.array([0.3, 0.7], dtype=np.float32)

    island_reg = NEATRegressor(
        pop_size=10,
        max_generations=6,
        engine="island",
        engine_params={"num_islands": 2, "migration_interval": 3},
        random_state=42,
    )
    island_reg.fit(X, y)
    assert hasattr(island_reg, "best_genome_")
    preds = island_reg.predict(X)
    assert preds.shape == (2,)
