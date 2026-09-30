"""Shared evaluator resolution for all backend providers.

Consolidates the 4 duplicated ``isinstance(fitness_spec, str)`` → parse →
build evaluator patterns from ``factory.py`` into a single function.

This module also fixes the known bug where ``build_composable_evosax_engine``
was missing BBOB numeric-index handling (``bbob:fn=3`` would fail).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def resolve_evaluator_base(
    fitness_spec: Any,
    *,
    maximize: bool = False,
    seed: int = 42,
    num_dims: int = 10,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    data_registry: Optional[Dict[str, Any]] = None,
) -> Any:
    """Resolve a fitness spec into a concrete evaluator object.

    Handles all input forms:

    - **String specs**: Parsed via ``OperatorCatalog`` (e.g. ``"sphere:dim=5"``,
      ``"bbob:fn=3,dim=10"``).  BBOB numeric indices are resolved to function
      names.
    - **Dict specs**: Parsed via ``OperatorCatalog``.
    - **Pre-built objects**: Returned as-is.
    - **None**: Returns a default Sphere evaluator.

    Parameters
    ----------
    fitness_spec : str, dict, evaluator object, or None
        The fitness function specification.
    maximize : bool
        Optimization direction.
    seed : int
        Random seed for evaluator construction (e.g. BBOB instance seed).
    num_dims : int
        Number of dimensions (used as fallback if not in spec).
    bounds : tuple of float
        Search space bounds (used as fallback).
    data_registry : dict, optional
        Resolved data registry for data-dependent evaluators.

    Returns
    -------
    evaluator
        A concrete evaluator object.
    """
    from malthusjax.composer.catalog import OperatorCatalog

    if fitness_spec is None:
        # Default: Sphere evaluator via catalog
        return OperatorCatalog().get(
            f"sphere:dim={num_dims},maximize={maximize},seed={seed}",
            data_registry=data_registry,
        )

    if isinstance(fitness_spec, dict):
        spec_dict = fitness_spec.copy()
        if "seed" not in spec_dict:
            spec_dict["seed"] = seed
        return OperatorCatalog().get(
            spec_dict,
            data_registry=data_registry,
            maximize=maximize,
        )

    if isinstance(fitness_spec, str):
        return _resolve_string_spec(
            fitness_spec,
            maximize=maximize,
            seed=seed,
            num_dims=num_dims,
            bounds=bounds,
            data_registry=data_registry,
        )

    # Pre-built evaluator object — return as-is
    return fitness_spec


def _resolve_string_spec(
    fitness_spec: str,
    *,
    maximize: bool,
    seed: int,
    num_dims: int,
    bounds: Tuple[float, float],
    data_registry: Optional[Dict[str, Any]],
) -> Any:
    """Resolve a string fitness specification.

    Handles BBOB numeric-index resolution (the logic that was missing in the
    composable evosax builder) and delegates to ``OperatorCatalog`` for all
    other specs.
    """
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
    from malthusjax.core.fitness.composable.environments import BBOBEnv
    from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
    from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

    cat = OperatorCatalog()
    parsed_name, parsed_params = cat.parse_spec(fitness_spec)

    # Inject seed if not present in the spec
    resolved_seed = parsed_params.get("seed", seed)

    if parsed_name == "bbob":
        return _resolve_bbob_spec(
            parsed_params,
            maximize=maximize,
            seed=resolved_seed,
            num_dims=num_dims,
        )

    # For all other string specs, use the catalog.
    # Inject seed into the spec string if missing.
    spec_str = fitness_spec
    if "seed=" not in spec_str:
        if ":" in spec_str:
            spec_str = f"{spec_str},seed={seed}"
        else:
            spec_str = f"{spec_str}:seed={seed}"

    return cat.get(spec_str, data_registry=data_registry, maximize=maximize)


def _resolve_bbob_spec(
    parsed_params: Dict[str, Any],
    *,
    maximize: bool,
    seed: int,
    num_dims: int,
) -> Any:
    """Resolve a BBOB fitness spec, including numeric function indices.

    This consolidates the BBOB-specific logic from ``build_evosax_engine``
    (factory.py L175-202) and fixes the gap where
    ``build_composable_evosax_engine`` was missing the integer-index branch.
    """
    from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
    from malthusjax.core.fitness.composable.environments import BBOBEnv
    from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
    from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

    fn_param = parsed_params.get("fn_name", parsed_params.get("fn", None))

    if fn_param is None:
        import evosax.problems.bbob.meta_bbob as mb

        bbob_keys = list(mb.bbob_fns.keys())
        for key in list(parsed_params.keys()):
            for fn in bbob_keys:
                if fn in key.lower():
                    fn_param = fn
                    if ":" in key:
                        sub_key = key.split(":")[-1]
                        parsed_params[sub_key] = parsed_params[key]
                    break
            if fn_param is not None:
                break

    if isinstance(fn_param, int):
        # Numeric BBOB index — resolve to function name
        import evosax.problems.bbob.meta_bbob as mb

        bbob_keys = list(mb.bbob_fns.keys())
        if fn_param < 1 or fn_param > len(bbob_keys):
            raise ValueError(
                f"BBOB function index {fn_param} is out of range (1-{len(bbob_keys)})"
            )
        fn_name = bbob_keys[fn_param - 1]
    elif fn_param is None:
        raise ValueError(
            "BBOB fitness specification requires either fn_name or fn index"
        )
    else:
        fn_name = fn_param

    dims = parsed_params.get("dim", parsed_params.get("num_dims", num_dims))
    resolved_maximize = parsed_params.get("maximize", maximize)
    resolved_seed = parsed_params.get("seed", seed)

    return OptimizationEvaluator(
        env=BBOBEnv.create(fn_name=fn_name, num_dims=dims, seed=resolved_seed),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=resolved_maximize),
    )
