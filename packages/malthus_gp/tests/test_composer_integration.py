"""Tests for Malthus-GP Composer and EngineRegistry integration."""

import numpy as np

from malthus_gp import (
    CGPClassifier,
    CGPRegressor,
    DCGPRegressor,
    SymbolicFeatureTransformer,
)
from malthusjax.composer.engine_catalog import EngineRegistry


def test_engine_registry_contains_cgp_and_island() -> None:
    """Verifies that 'cgp' and 'island' engines are registered in EngineRegistry."""
    registry = EngineRegistry()
    available = registry.list_available()
    assert "cgp" in available
    assert "island" in available


def test_cgp_regressor_compose_specs() -> None:
    """Tests dynamic Composer string specification composition for CGPRegressor."""
    reg = CGPRegressor(
        n_rows=2,
        n_cols=25,
        pop_size=5,
        max_generations=50,
        engine="cgp",
    )
    specs = reg.compose_specs(n_inputs=3, n_outputs=1)
    assert "genome" in specs
    assert "selection" in specs
    assert "crossover" in specs
    assert "mutation" in specs
    assert "engine" in specs

    assert "cartesian_genome:num_rows=2,num_cols=25,num_inputs=3,num_outputs=1" in specs["genome"]
    assert "cgp_neutral" in specs["selection"]
    assert specs["crossover"] == "cartesian_noop"
    assert "cartesian_micro" in specs["mutation"]
    assert specs["engine"] == "cgp:pop_size=5,generations=50"


def test_cgp_regressor_island_engine() -> None:
    """Tests fitting CGPRegressor using distributed island model."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = X[:, 0] ** 2 + 0.1

    reg = CGPRegressor(
        n_rows=1,
        n_cols=10,
        pop_size=5,
        max_generations=20,
        engine="island",
        engine_params={"num_islands": 2, "migration_interval": 10, "num_migrants": 1},
        random_state=42,
    )
    reg.fit(X, y)

    assert hasattr(reg, "best_genome_")
    assert hasattr(reg, "best_fitness_")
    assert isinstance(reg.best_fitness_, float)
    assert not np.isnan(reg.best_fitness_)

    preds = reg.predict(X)
    assert preds.shape == (20,)
    assert not np.isnan(preds).any()


def test_dcgp_regressor_composer() -> None:
    """Tests DCGPRegressor integration with Composer."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = np.sin(X[:, 0])

    dcgp = DCGPRegressor(
        n_rows=1,
        n_cols=8,
        pop_size=5,
        max_generations=10,
        lsmf_epochs=2,
        engine="cgp",
        random_state=42,
    )
    dcgp.fit(X, y)

    assert hasattr(dcgp, "best_genome_")
    assert hasattr(dcgp, "best_fitness_")
    formula = dcgp.formula()
    assert isinstance(formula, str)
    assert len(formula) > 0


def test_cgp_classifier_composer() -> None:
    """Tests CGPClassifier integration with Composer."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = (X[:, 0] > 0.0).astype(int)

    clf = CGPClassifier(
        n_rows=1,
        n_cols=8,
        pop_size=5,
        max_generations=10,
        engine="cgp",
        random_state=42,
    )
    clf.fit(X, y)

    assert hasattr(clf, "best_genome_")
    probs = clf.predict_proba(X)
    assert probs.shape == (20, 2)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_symbolic_transformer_composer() -> None:
    """Tests SymbolicFeatureTransformer integration with Composer."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = X[:, 0] ** 2

    trans = SymbolicFeatureTransformer(
        n_features=2,
        n_rows=1,
        n_cols=8,
        pop_size=5,
        max_generations=10,
        engine="cgp",
        random_state=42,
    )
    trans.fit(X, y)

    Xt = trans.transform(X)
    assert Xt.shape == (20, 2)
    assert not np.isnan(Xt).any()
