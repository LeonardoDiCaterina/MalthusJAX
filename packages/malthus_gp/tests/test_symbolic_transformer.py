"""Tests for SymbolicFeatureTransformer and Scikit-Learn Pipeline integration."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from malthus_gp.learn import SymbolicFeatureTransformer


def test_transformer_contract_fit_transform():
    """Verify fit, transform, fit_transform, shapes, and formulas."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-2, 2, size=(30, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2])).astype(np.float32)

    transformer = SymbolicFeatureTransformer(
        n_features=4,
        include_original=False,
        max_generations=20,
        random_state=42,
    )
    transformer.fit(X, y)

    X_trans = transformer.transform(X)
    assert X_trans.shape == (30, 4)
    assert np.all(np.isfinite(X_trans))

    # Test fit_transform equivalence
    X_fit_trans = transformer.fit_transform(X, y)
    assert X_fit_trans.shape == (30, 4)
    assert np.all(np.isfinite(X_fit_trans))

    # Test formulas
    formulas = transformer.to_formula()
    assert len(formulas) == 4
    for f in formulas:
        assert isinstance(f, str)
        assert len(f) > 0

    # Test feature names out
    feat_names = transformer.get_feature_names_out()
    assert feat_names == ["cgp_feat_0", "cgp_feat_1", "cgp_feat_2", "cgp_feat_3"]


def test_transformer_include_original():
    """Verify include_original=True appends engineered features to raw inputs."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, size=(25, 2)).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).astype(np.float32)

    transformer = SymbolicFeatureTransformer(
        n_features=3,
        include_original=True,
        max_generations=20,
        random_state=42,
    )
    X_out = transformer.fit_transform(X, y)

    # Output shape should be (25, 2 + 3) = (25, 5)
    assert X_out.shape == (25, 5)
    np.testing.assert_allclose(X_out[:, :2], X)

    names = transformer.get_feature_names_out()
    assert names == ["x0", "x1", "cgp_feat_0", "cgp_feat_1", "cgp_feat_2"]


def test_transformer_sklearn_pipeline_integration():
    """Verify SymbolicFeatureTransformer executes seamlessly in a scikit-learn Pipeline."""
    rng = np.random.RandomState(42)
    X_train = rng.uniform(-1.5, 1.5, size=(40, 2)).astype(np.float32)
    y_train = (X_train[:, 0] * X_train[:, 1] + 0.5 * X_train[:, 0]).astype(np.float32)
    X_test = rng.uniform(-1.5, 1.5, size=(20, 2)).astype(np.float32)

    pipeline = Pipeline(
        [
            (
                "sym_features",
                SymbolicFeatureTransformer(n_features=3, max_generations=30, random_state=42),
            ),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))


def test_transformer_boosts_linear_model():
    """Verify that automated symbolic feature synthesis improves linear model capacity on non-linear data."""
    rng = np.random.RandomState(42)
    # Non-linear interaction: y = x0 * x1 + x0^2
    X = rng.uniform(-2, 2, size=(60, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 0] ** 2).astype(np.float32)

    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    # Baseline: Plain Linear Regression on raw features (cannot represent x0 * x1 or x0^2)
    raw_lr = LinearRegression().fit(X_train, y_train)
    raw_r2 = raw_lr.score(X_test, y_test)

    # Boosted: Pipeline with SymbolicFeatureTransformer (synthesizes non-linear interactions)
    sym_pipe = Pipeline(
        [
            (
                "sym",
                SymbolicFeatureTransformer(
                    n_features=5, include_original=True, max_generations=100, random_state=42
                ),
            ),
            ("lr", LinearRegression()),
        ]
    )
    sym_pipe.fit(X_train, y_train)
    boosted_r2 = sym_pipe.score(X_test, y_test)

    assert np.isfinite(boosted_r2)
    # The symbolic features should capture non-linear patterns, outperforming raw linear regression
    assert boosted_r2 > raw_r2


def test_transformer_grid_search_cv():
    """Verify GridSearchCV can tune SymbolicFeatureTransformer hyperparameters within a Pipeline."""
    from sklearn.model_selection import GridSearchCV

    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, size=(30, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1]).astype(np.float32)

    pipe = Pipeline(
        [
            ("sym", SymbolicFeatureTransformer(max_generations=15, random_state=42)),
            ("ridge", Ridge()),
        ]
    )

    param_grid = {
        "sym__n_features": [2, 4],
        "ridge__alpha": [0.1, 1.0],
    }

    grid = GridSearchCV(pipe, param_grid, cv=2)
    grid.fit(X, y)

    assert grid.best_estimator_ is not None
    preds = grid.predict(X)
    assert preds.shape == (30,)
    assert np.all(np.isfinite(preds))
