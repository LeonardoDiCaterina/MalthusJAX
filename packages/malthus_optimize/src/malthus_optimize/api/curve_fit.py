"""Bounded global optimization API curve_fit powered by global optimization and JAX residual evaluators."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from malthus_optimize.composer.bridge import run_composer_optimization
from malthus_optimize.core.bounds import ParsedBounds, parse_bounds
from malthus_optimize.evaluators.least_squares_evaluator import make_least_squares_evaluator


def curve_fit(
    f: Callable[..., Any],
    xdata: Any,
    ydata: Any,
    p0: Optional[Sequence[float]] = None,
    sigma: Optional[Any] = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: Union[Sequence[Tuple[float, float]], Any] = (-np.inf, np.inf),
    method: str = "de",
    jac: Any = None,
    *,
    full_output: bool = False,
    maxfev: Optional[int] = None,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Fits a non-linear model function f to data (xdata, ydata) using global optimization.

    Limited alternative to :func:`scipy.optimize.curve_fit`, avoiding local minima
    in peak fitting, spectroscopy, and kinetic modeling.

    Parameters
    ----------
    f : callable
        The model function, ``f(x, ...)``. It must take the independent
        variable as the first argument and the parameters to fit as separate remaining arguments.
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data: nominal values measured at ``xdata``.
    p0 : array_like, optional
        Initial guess for the parameters.
    sigma : array_like, optional
        Determines the uncertainty in ``ydata``.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters.
    method : str, optional
        Global optimization backend: ``'de'`` (default), ``'cmaes'``, or ``'pso'``.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared residuals
        is minimized.
    pcov : 2-D array
        The estimated covariance of popt.
    """
    x_arr = np.asarray(xdata, dtype=float)
    y_arr = np.asarray(ydata, dtype=float)

    if jac is not None or full_output or maxfev is not None or not check_finite:
        raise NotImplementedError("jac, SciPy full_output, maxfev and disabled finite checks are unsupported")
    if set(kwargs) - {"pop_size","popsize","max_generations","maxiter","seed"}:
        raise NotImplementedError("Unsupported curve_fit options")
    if y_arr.ndim != 1 or x_arr.shape[-1] != y_arr.size or not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("Fitting data must be finite with aligned one-dimensional observations")
    if np.asarray(sigma).ndim == 2:
        raise NotImplementedError("Covariance-matrix sigma is unsupported")
    weights = np.ones(y_arr.size) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), y_arr.shape)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sigma must be finite and positive")
    # 1. Determine parameter dimensionality
    if p0 is not None:
        p0_arr = np.asarray(p0, dtype=np.float32)
        num_params = len(p0_arr)
    else:
        sig = inspect.signature(f)
        params = list(sig.parameters.keys())
        if len(params) > 1 and params[0] in ("x", "xdata", "t"):
            num_params = len(params) - 1
            p0_arr = np.ones(num_params, dtype=np.float32)
        else:
            num_params = 1
            p0_arr = np.ones(1, dtype=np.float32)

    # 2. Parse bounds
    has_finite_bounds = False
    if bounds is not None:
        if (
            isinstance(bounds, (list, tuple))
            and len(bounds) == 2
            and not isinstance(bounds[0], (list, tuple, np.ndarray))
        ):
            # Single (min, max) pair
            if not (np.isneginf(bounds[0]) and np.isposinf(bounds[1])):
                has_finite_bounds = True
        else:
            has_finite_bounds = True

    if has_finite_bounds:
        parsed_bounds = parse_bounds(bounds, num_dims=num_params)
    else:
        # Default domain centered around p0
        span = np.maximum(np.abs(p0_arr) * 4.0, 10.0)
        parsed_bounds = ParsedBounds(
            lower=p0_arr - span,
            upper=p0_arr + span,
            num_dims=num_params,
        )

    # 3. Configure iterations and population
    pop_size = int(kwargs.get("pop_size", kwargs.get("popsize", max(20, num_params * 10))))
    max_generations = int(kwargs.get("max_generations", kwargs.get("maxiter", 80)))
    seed = int(kwargs.get("seed", 42))

    # 4. Construct evaluator
    evaluator = make_least_squares_evaluator(
        model_func=f,
        xdata=x_arr,
        ydata=y_arr,
        sigma=sigma,
        lower=parsed_bounds.lower,
        upper=parsed_bounds.upper,
    )

    # 5. Run global optimization via Composer
    popt, best_loss, nfev, nit, history = run_composer_optimization(
        evaluator=evaluator,
        bounds=parsed_bounds,
        method=method,
        pop_size=pop_size,
        generations=max_generations,
        seed=seed,
        use_python_loop=False,
    )

    # Whiten both residuals and Jacobian before covariance estimation.
    popt = np.asarray(popt, dtype=float)
    if not np.isfinite(popt).all():
        raise ValueError("Backend returned nonfinite fitted parameters")
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(popt))
    columns = []
    for j, step in enumerate(steps):
        plus, minus = popt.copy(), popt.copy()
        plus[j] += step
        minus[j] -= step
        columns.append((np.asarray(f(x_arr,*plus),dtype=float)-np.asarray(f(x_arr,*minus),dtype=float))/(2*step))
    if evaluator.config.is_pure_jax:
        import jax
        import jax.numpy as jnp
        jacobian = np.asarray(jax.jacfwd(lambda p: f(jnp.asarray(x_arr), *p))(jnp.asarray(popt)), dtype=float)
    else:
        jacobian = np.column_stack(columns)
    J = jacobian / weights[:,None]
    residual = (np.asarray(f(x_arr,*popt),dtype=float) - y_arr) / weights
    if not np.isfinite(J).all() or not np.isfinite(residual).all():
        raise ValueError("Nonfinite residuals or Jacobian")
    dof = len(y_arr) - num_params
    if np.linalg.matrix_rank(J) < num_params or (not absolute_sigma and dof <= 0):
        pcov = np.full((num_params,num_params),np.inf)
    else:
        pcov = np.linalg.pinv(J.T @ J)
        if not absolute_sigma:
            pcov *= float(residual @ residual) / dof

    return popt, pcov


__all__ = ["curve_fit"]
