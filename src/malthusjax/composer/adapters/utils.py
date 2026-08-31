import warnings
from typing import Any, Optional, Tuple


def resolve_bounds(
    bounds: Optional[Tuple[float, float]],
    evaluator: Any,
    *,
    caller_name: str,
    default: Tuple[float, float] = (-5.0, 5.0),
) -> Tuple[float, float]:
    """Resolves bounds from explicit arguments or evaluator config."""
    if bounds is not None:
        return bounds

    if (
        evaluator is not None
        and hasattr(evaluator, "config")
        and hasattr(evaluator.config, "genome_config")
        and hasattr(evaluator.config.genome_config, "bounds")
        and evaluator.config.genome_config.bounds is not None
    ):
        return evaluator.config.genome_config.bounds

    framework_name = caller_name.split("_")[1] if len(caller_name.split("_")) > 1 else caller_name
    warnings.warn(
        f"No bounds were explicitly provided to `{caller_name}`, and the evaluator "
        "did not provide a `genome_config` with bounds. Falling back to the default "
        f"bounds of {default} for {framework_name} initialization. "
        f"To change this, either pass `bounds=(min, max)` to `{caller_name}`, "
        "or specify bounds in your TOML config under the genome section."
    )
    return default
