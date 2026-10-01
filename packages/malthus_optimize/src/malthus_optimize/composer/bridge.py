"""Bridge between SciPy optimization calls and MalthusJAX Composer engines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import numpy as np

from malthus_optimize.core.bounds import ParsedBounds
from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_optimize.composer.bridge")

METHOD_TO_STRATEGY: Dict[str, str] = {
    "de": "DifferentialEvolution",
    "differential_evolution": "DifferentialEvolution",
    "cmaes": "CMA_ES",
    "cma_es": "CMA_ES",
    "pso": "PSO",
    "ga": "SimpleGA",
    "simple_ga": "SimpleGA",
    "open_es": "Open_ES",
    "xnes": "xNES",
    "snes": "SNES",
    "pgpe": "PGPE",
    "ars": "ARS",
}


def run_composer_optimization(
    evaluator: Any,
    bounds: ParsedBounds,
    method: str = "de",
    pop_size: int = 50,
    generations: int = 100,
    seed: int = 42,
    strategy_params: Optional[Dict[str, Any]] = None,
    use_python_loop: bool = False,
) -> Tuple[np.ndarray, float, int, int, list[dict[str, Any]]]:
    """Executes global optimization using MalthusJAX Composer.

    Args:
        evaluator: Instantiated ScipyObjectiveEvaluator or LeastSquaresEvaluator.
        bounds: Parsed search domain bounds.
        method: Evolutionary strategy identifier.
        pop_size: Number of candidate solutions per generation.
        generations: Maximum iteration count.
        seed: PRNG seed.
        strategy_params: Optional strategy hyperparameters (e.g. differential_weight, crossover_rate).
        use_python_loop: Whether to run outside lax.scan for un-jittable functions.

    Returns:
        (best_x_physical, best_fitness, total_evaluations, actual_iterations, history)
    """
    strategy_name = METHOD_TO_STRATEGY.get(method.lower(), method)

    # 1. Build adapted engine via Composer
    adapted = build_evosax_engine(
        strategy_name=strategy_name,
        evaluator=evaluator,
        pop_size=pop_size,
        generations=generations,
        num_dims=bounds.num_dims,
        bounds=(0.0, 1.0),
        maximize=False,
        seed=seed,
        strategy_params=strategy_params,
        use_python_loop=use_python_loop,
    )

    eng = adapted.engine
    key = jax.random.PRNGKey(seed)
    key_run, key_init = jax.random.split(key)

    # 2. Initialize and compile scan loop
    state_init = eng.init_fn(eng.framework_obj, key_init, eng.framework_params, None)
    if use_python_loop:
        run_loop = eng._build_python_loop()
    else:
        run_loop = eng._build_jit_loop()

    # 3. Execute
    final_state, scan_history = run_loop(key_run, state_init)

    # 4. Extract best solution
    if hasattr(final_state, "best_solution"):
        raw_best_norm = np.asarray(final_state.best_solution)
    elif hasattr(final_state, "best_member"):
        raw_best_norm = np.asarray(final_state.best_member)
    elif hasattr(final_state, "mean"):
        raw_best_norm = np.asarray(final_state.mean)
    else:
        raw_best_norm = np.full(bounds.num_dims, 0.5, dtype=np.float32)

    # 5. Denormalize from [0, 1] to physical bounds
    best_x = bounds.denormalize(np.clip(raw_best_norm, 0.0, 1.0))
    best_fun = float(final_state.best_fitness)

    total_evals = pop_size * generations
    nit = generations

    # Build metric history
    history: list[dict[str, Any]] = []
    if "best_fitness" in scan_history:
        fits = np.asarray(scan_history["best_fitness"])
        for g, fit in enumerate(fits):
            history.append({"generation": g + 1, "best_fitness": float(fit)})

    return best_x, best_fun, total_evals, nit, history


__all__ = ["run_composer_optimization", "METHOD_TO_STRATEGY"]
