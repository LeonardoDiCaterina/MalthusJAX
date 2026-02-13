from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

# Annotate as Any so mypy does not complain about dynamic loader
toml: Any = _toml


def load_config(path: str, pipeline_name: str) -> Dict[str, Any]:
    """Very small TOML loader that returns the raw pipeline section.
    (Keep this lightweight—validation and pydantic will come later.)
    """
    try:
        with open(path, "rb") as f:
            cfg = toml.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e
    pipelines = cfg.get("pipelines", {})
    pipeline = pipelines.get(pipeline_name)
    if pipeline is None:
        raise KeyError(f"Pipeline '{pipeline_name}' not found in {path}")
    return cast(Dict[str, Any], pipeline)


# ---------------------------------------------------------------------------
# Experiment-level config loader (Composer TOML schema)
# ---------------------------------------------------------------------------

# Keys that live under [experiment] and are NOT forwarded as quick_run kwargs.
_EXPERIMENT_META_KEYS = {"name", "output_dir"}


def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load a Composer-style experiment TOML.

    Expected schema::

        [experiment]
        name = "my_experiment"
        output_dir = "results/my_experiment"

        [experiment.shared]          # defaults inherited by all pipelines
        fitness      = "sphere:dim=10"
        pop_size     = 50
        generations  = 100
        genome_length = 10
        bounds       = [-5.0, 5.0]
        seeds        = [42, 43, 44]
        prng_impl    = "threefry"

        [pipelines.blend_ga]
        backend   = "malthusjax"
        crossover = "blend:alpha=0.5"
        mutation  = "gaussian:mutation_rate=0.1"

        [pipelines.evosax_simple]
        backend         = "evosax"
        evosax_strategy = "SimpleGA"

    Parameters
    ----------
    path
        Path to the TOML file.
    pipelines
        Optional list of pipeline names to load.  If ``None``, all
        ``[pipelines.*]`` sections are returned.

    Returns
    -------
    (experiment_meta, resolved_pipelines)
        *experiment_meta* contains top-level metadata (``name``,
        ``output_dir``) plus a ``"shared"`` key with the merged defaults.

        *resolved_pipelines* is ``{name: kwargs_dict}`` where each dict
        is ready to be splatted into :meth:`Composer.quick_run`.  Shared
        defaults are already merged (pipeline-level keys win).
    """
    try:
        with open(path, "rb") as f:
            cfg = toml.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e

    # ---- experiment section ---------------------------------------------
    experiment_raw: Dict[str, Any] = cfg.get("experiment", {})
    shared: Dict[str, Any] = dict(experiment_raw.get("shared", {}))

    # TOML arrays → Python tuples where quick_run expects them
    if "bounds" in shared and isinstance(shared["bounds"], list):
        shared["bounds"] = tuple(shared["bounds"])
    if "seeds" in shared and isinstance(shared["seeds"], list):
        shared["seeds"] = tuple(shared["seeds"])

    experiment_meta: Dict[str, Any] = {
        k: v for k, v in experiment_raw.items() if k in _EXPERIMENT_META_KEYS
    }
    experiment_meta["shared"] = shared

    # ---- pipelines section ----------------------------------------------
    raw_pipelines: Dict[str, Any] = cfg.get("pipelines", {})
    if not raw_pipelines:
        raise KeyError(f"No [pipelines.*] sections found in {path}")

    # Filter if caller requested specific pipelines
    if pipelines is not None:
        missing = set(pipelines) - set(raw_pipelines)
        if missing:
            raise KeyError(
                f"Pipelines not found in {path}: {sorted(missing)}. "
                f"Available: {sorted(raw_pipelines.keys())}"
            )
        raw_pipelines = {k: raw_pipelines[k] for k in pipelines}

    # Merge shared → pipeline (pipeline wins)
    resolved: Dict[str, Dict[str, Any]] = {}
    for name, pipeline_cfg in raw_pipelines.items():
        merged = {**shared, **pipeline_cfg}
        # Convert TOML list → tuple for bounds/seeds at pipeline level too
        if "bounds" in merged and isinstance(merged["bounds"], list):
            merged["bounds"] = tuple(merged["bounds"])
        if "seeds" in merged and isinstance(merged["seeds"], list):
            merged["seeds"] = tuple(merged["seeds"])
        # Use pipeline name as experiment_name unless explicitly set
        merged.setdefault("experiment_name", name)
        resolved[name] = merged

    return experiment_meta, resolved
