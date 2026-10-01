import numpy as np
from malthus_gp.learn import CGPClassifier, CGPRegressor


def test_cgp_regressor_fit_predict():
    # Simple linear target: y = x0 + x0 = 2*x0
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2.0 * X.flatten()

    reg = CGPRegressor(
        n_rows=1,
        n_cols=10,
        levels_back=-1,
        pop_size=5,
        max_generations=50,
        random_state=0,
    )
    reg.fit(X, y)

    # Check fitted attributes
    assert hasattr(reg, "best_genome_")
    assert hasattr(reg, "best_fitness_")
    assert reg.best_fitness_ >= 0.0

    # Check predictions
    preds = reg.predict(X)
    assert preds.shape == (20,)

    # Check formula
    formula = reg.to_formula()
    assert isinstance(formula, str)
    assert len(formula) > 0

    # Check active mask
    mask = reg.get_active_mask()
    assert mask.ndim == 1
    assert mask.dtype == bool or mask.dtype == np.bool_

    # Check get_params / set_params
    params = reg.get_params()
    assert params["n_cols"] == 10
    reg.set_params(n_cols=20)
    assert reg.n_cols == 20


def test_cgp_quartic_polynomial_benchmark():
    """Validates Cartesian GP on the classic quartic benchmark from Miller & Thomson."""
    # Target: y = x^4 + x^3 + x^2 + x
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = (X**4 + X**3 + X**2 + X).flatten()

    reg = CGPRegressor(
        n_rows=1,
        n_cols=30,
        levels_back=-1,
        pop_size=5,
        max_generations=150,
        random_state=42,
    )
    reg.fit(X, y)

    preds = reg.predict(X)
    mse = np.mean((y - preds) ** 2)
    assert mse < 10.0  # Evolution is making progress within 150 generations


def test_cgp_classifier_fit_predict():
    # Simple 2-class separation: x0 > 0 -> class 1, else class 0
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = CGPClassifier(
        n_rows=1,
        n_cols=10,
        pop_size=5,
        max_generations=50,
        random_state=1,
    )
    clf.fit(X, y)

    # Test predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    # Test predict
    preds = clf.predict(X)
    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})
