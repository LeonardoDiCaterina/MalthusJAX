"""Shared initial population generation for evolutionary backends.

Eliminates duplicate generate_initial_population() implementations across
providers, supporting deterministic initial populations for compare() parity.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.random as jr


def generate_initial_population(
    config: Any,
    pop_seed: int,
) -> Optional[Any]:
    """Generate deterministic initial population for compare() parity.

    Supports both ExperimentConfig instances and legacy dicts.

    Handles:
    - BBOB problem-specific sampling (via bbob_eval.env._problem.sample)
    - Uniform random fallback within bounds
    - Returns None for backends that manage their own initialization
      (e.g., TensorNEAT returning a tuple of nodes and connections).
    """
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.composer.config import infer_genome_length
    from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
    from malthusjax.core.fitness.composable.environments import BBOBEnv
    from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
    from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

    if hasattr(config, "population") and hasattr(config, "execution"):
        pop_size = int(config.population.size)
        genome_length = int(config.population.genome_length)
        bounds = tuple(config.population.bounds)
        fitness_spec = getattr(config.backend, "fitness", None)
        maximize = bool(config.execution.maximize)
    elif isinstance(config, dict):
        pop_size = int(config.get("pop_size", 50))
        genome_length = int(infer_genome_length(config))
        raw_bounds = config.get("bounds", (-5.0, 5.0))
        bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
        fitness_spec = config.get("fitness")
        maximize = bool(config.get("maximize", False))
    else:
        return None

    if fitness_spec and isinstance(fitness_spec, str) and "bbob" in fitness_spec.lower():
        cat = OperatorCatalog()
        parsed_name, parsed_params = cat.parse_spec(fitness_spec)
        if parsed_name == "bbob":
            fn = parsed_params.get("fn_name", parsed_params.get("fn", "rosenbrock"))
            dims = parsed_params.get("dim", parsed_params.get("num_dims", genome_length))
            bbob_seed = parsed_params.get("seed", 0)
            bbob_eval = OptimizationEvaluator(
                env=BBOBEnv.create(fn_name=fn, num_dims=dims, seed=bbob_seed),
                transform=IdentityTransform(),
                interpreter=IdentityInterpreter(),
                output=ScalarOutput(maximize=maximize),
            )
            pop_key = jr.PRNGKey(pop_seed)
            sample_keys = jr.split(pop_key, pop_size)
            return jax.vmap(bbob_eval.env._problem.sample)(sample_keys)

    return jr.uniform(
        jr.PRNGKey(pop_seed),
        (pop_size, genome_length),
        minval=float(bounds[0]),
        maxval=float(bounds[1]),
    )
