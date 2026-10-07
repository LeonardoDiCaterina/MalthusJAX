"""SciPy drop-in minimize function supporting multiple global optimization strategies."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from malthus_optimize.composer.bridge import run_composer_optimization
from malthus_optimize.core.bounds import ParsedBounds, parse_bounds
from malthus_optimize.core.result import OptimizeResult
from malthus_optimize.evaluators.scipy_evaluator import make_scipy_evaluator


def minimize(
    fun: Callable[..., Any],
    x0: Any = None,
    args: Tuple[Any, ...] = (),
    method: Optional[str] = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Union[Sequence[Tuple[float, float]], Any] = None,
    constraints: Any = (),
    tol: Optional[float] = None,
    callback: Optional[Callable[..., Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> OptimizeResult:
    """Minimization of scalar function of one or more variables.

    Drop-in replacement for :func:`scipy.optimize.minimize`, dispatching global
    evolutionary and swarm strategies through MalthusJAX Composer.

    Supported methods:
    - ``'de'`` / ``'differential_evolution'``
    - ``'cmaes'`` / ``'cma_es'`` (Covariance Matrix Adaptation Evolution Strategy)
    - ``'pso'`` (Particle Swarm Optimization)
    - ``'ga'`` / ``'simple_ga'`` (Genetic Algorithm)
    - ``'open_es'`` / ``'xnes'`` / ``'pgpe'``

    Parameters
    ----------
    fun : callable
        Objective function: ``fun(x, *args) -> float``.
    x0 : ndarray, optional
        Initial guess parameter vector.
    args : tuple, optional
        Extra arguments passed to ``fun``.
    method : str, optional
        Optimization method name. Defaults to ``'cmaes'`` if x0 is provided, or ``'de'``.
    bounds : sequence or Bounds, optional
        Bounds on variables.
    tol : float, optional
        Tolerance for termination.
    callback : callable, optional
        Called after each iteration as ``callback(xk)``.
    options : dict, optional
        A dictionary of solver options:
        - ``maxiter``: Maximum generations (default: 100).
        - ``popsize`` / ``pop_size``: Population size (default: 50).
        - ``seed``: Random seed.
        - ``disp``: Boolean display flag.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a :class:`scipy.optimize.OptimizeResult`.
    """
    opts = options or {}
    maxiter = int(opts.get("maxiter", opts.get("max_generations", 100)))
    pop_size = int(opts.get("popsize", opts.get("pop_size", 50)))
    seed = int(opts.get("seed", 42))
    disp = bool(opts.get("disp", False))

    # 1. Resolve strategy method
    if method is None:
        method = "cmaes" if x0 is not None else "de"
    method_key = method.lower()

    # 2. Resolve bounds
    if bounds is not None:
        parsed_bounds = parse_bounds(bounds)
    elif x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.float32)
        span = np.maximum(np.abs(x0_arr) * 2.0, 5.0)
        lb = x0_arr - span
        ub = x0_arr + span
        parsed_bounds = ParsedBounds(lower=lb, upper=ub, num_dims=len(x0_arr))
    else:
        raise ValueError(
            "Must provide either 'bounds' or 'x0' to determine parameter search dimensionality."
        )

    # 3. Build evaluator
    evaluator = make_scipy_evaluator(
        func=fun,
        args=args,
        lower=parsed_bounds.lower,
        upper=parsed_bounds.upper,
        maximize=False,
    )

    if disp:
        print(
            f"Malthus-Optimize minimize(method='{method}'): {parsed_bounds.num_dims} dims, {pop_size} pop, {maxiter} iters"
        )

    # 4. Execute via Composer bridge
    best_x, best_fun, nfev, nit, history = run_composer_optimization(
        evaluator=evaluator,
        bounds=parsed_bounds,
        method=method_key,
        pop_size=pop_size,
        generations=maxiter,
        seed=seed,
        use_python_loop=False,
    )

    if disp:
        print(f"Optimization finished! Best fitness: {best_fun:.6e}")

    if callback is not None:
        try:
            callback(best_x)
        except Exception:
            pass

    return OptimizeResult(
        x=best_x,
        fun=best_fun,
        success=True,
        status=0,
        message=f"Optimization terminated successfully via {method.upper()}.",
        nit=nit,
        nfev=nfev,
        history=history,
    )


__all__ = ["minimize"]
