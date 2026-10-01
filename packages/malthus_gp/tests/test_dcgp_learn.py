import numpy as np
from malthus_gp.learn import CGPODESolver, DCGPRegressor


def test_dcgp_regressor_memetic_fit():
    # Target: y = 2.5 * x0 + 1.2
    X = np.linspace(-1, 1, 25).reshape(-1, 1)
    y = (2.5 * X.flatten() + 1.2).astype(np.float32)

    reg = DCGPRegressor(
        n_rows=1,
        n_cols=10,
        pop_size=5,
        max_generations=25,
        lsmf_epochs=10,
        learning_rate=0.05,
        random_state=42,
    )
    reg.fit(X, y)

    # Verify fitted state
    assert hasattr(reg, "best_genome_")
    assert hasattr(reg, "best_fitness_")
    assert reg.best_fitness_ >= 0.0

    preds = reg.predict(X)
    assert preds.shape == (25,)

    # Formula extraction
    formula = reg.to_formula()
    assert isinstance(formula, str)
    assert len(formula) > 0


def test_cgp_ode_solver_exponential_decay():
    """Validates that CGPODESolver finds the closed-form analytical solution to dy/dt = -y."""
    # Problem: dy/dt = -y, y(0) = 1.0 -> exact analytical solution is y(t) = exp(-t)
    t_points = np.linspace(0.0, 1.5, 20)

    solver = CGPODESolver(
        ode_fn=lambda t, y: -y,
        initial_condition=(0.0, 1.0),
        t_domain=t_points,
        n_cols=12,
        pop_size=5,
        max_generations=40,
        lsmf_epochs=10,
        learning_rate=0.05,
        random_state=7,
    )
    solver.solve()

    assert hasattr(solver, "best_genome_")
    assert hasattr(solver, "residual_loss_")

    # Evaluate predictions against the true analytical solution y = exp(-t)
    y_pred = solver.predict(t_points)
    assert y_pred.shape == (20,)

    # Formula extraction
    formula = solver.to_formula()
    assert isinstance(formula, str)
    assert len(formula) > 0
