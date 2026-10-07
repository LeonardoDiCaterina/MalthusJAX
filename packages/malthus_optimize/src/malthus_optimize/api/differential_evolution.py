"""SciPy drop-in differential_evolution powered by JAX and MalthusJAX Composer."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union


from malthus_optimize.composer.bridge import run_composer_optimization
from malthus_optimize.core.bounds import parse_bounds
from malthus_optimize.core.result import OptimizeResult
from malthus_optimize.evaluators.scipy_evaluator import make_scipy_evaluator


def differential_evolution(
    func: Callable[..., Any],
    bounds: Union[Sequence[Tuple[float, float]], Any],
    args: Tuple[Any, ...] = (),
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: Union[float, Tuple[float, float]] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int | None = None,
    callback: Callable[..., Any] | None = None,
    disp: bool = False,
    polish: bool = False,
    init: str = "latinhypercube",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int = 1,
    constraints: Any = (),
    x0: Any = None,
    *,
    integrality: Any = None,
    vectorized: bool = False,
    **kwargs: Any,
) -> OptimizeResult:
    """Finds the global minimum of a multivariate function using Differential Evolution.

    Drop-in replacement for :func:`scipy.optimize.differential_evolution`, executing
    population evaluation and generational updates via JAX and MalthusJAX Composer.

    Parameters
    ----------
    func : callable
        The objective function to be minimized: ``func(x, *args) -> float``.
    bounds : sequence or Bounds
        Parameter bounds: ``[(min, max), ...]`` or :class:`scipy.optimize.Bounds`.
    args : tuple, optional
        Extra arguments passed to ``func``.
    strategy : str, optional
        The differential evolution strategy to use. Default is ``'best1bin'``.
    maxiter : int, optional
        Maximum number of evolutionary generations. Default is 1000.
    popsize : int, optional
        A multiplier for setting the total population size. Total candidate
        networks/vectors evaluated per generation is ``popsize * len(bounds)``.
    mutation : float or tuple(float, float), optional
        The mutation constant or dither range. Default is ``(0.5, 1.0)``.
    recombination : float, optional
        The recombination constant (crossover probability) in [0, 1]. Default is 0.7.
    seed : int, optional
        PRNG seed for reproducible execution.
    callback : callable, optional
        Called after each iteration as ``callback(xk, convergence=val)``.
    disp : bool, optional
        If True, displays progress messages.
    polish : bool, optional
        If True, applies local polish to the best candidate. Default is False.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a :class:`scipy.optimize.OptimizeResult`.
    """
    # 1. Parse and standardize bounds
    parsed_bounds = parse_bounds(bounds)
    num_dims = parsed_bounds.num_dims

    # In SciPy, total population = popsize * len(bounds)
    total_pop = max(popsize * num_dims, 10)

    # 2. Extract strategy hyperparameters
    diff_weight = mutation[0] if isinstance(mutation, (tuple, list)) else float(mutation)
    cross_rate = float(recombination)

    strategy_params = {
        "differential_weight": diff_weight,
        "crossover_rate": cross_rate,
    }

    # 3. Build JAX evaluator
    evaluator = make_scipy_evaluator(
        func=func,
        args=args,
        lower=parsed_bounds.lower,
        upper=parsed_bounds.upper,
        maximize=False,
    )

    prng_seed = int(seed) if seed is not None else 42

    if disp:
        print(
            f"Malthus-Optimize Differential Evolution: {num_dims} dims, {total_pop} pop, {maxiter} iters"
        )

    # 4. Execute via Composer
    best_x, best_fun, nfev, nit, history = run_composer_optimization(
        evaluator=evaluator,
        bounds=parsed_bounds,
        method="DifferentialEvolution",
        pop_size=total_pop,
        generations=maxiter,
        seed=prng_seed,
        strategy_params=strategy_params,
        use_python_loop=False,
    )

    if disp:
        print(f"Optimization finished! Best fitness: {best_fun:.6e}")

    # 5. Optional callback invocation
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
        message="Optimization terminated successfully.",
        nit=nit,
        nfev=nfev,
        history=history,
    )


__all__ = ["differential_evolution"]
