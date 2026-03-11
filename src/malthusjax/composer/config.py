"""Simple TOML-based configuration loader for Composer experiments.

This lightweight utility exposes ``load_config`` and
``load_experiment_config`` for parsing pipeline and experiment definitions
without pulling in heavier dependencies; validation and schema handling are
performed at call sites.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

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

_EXPERIMENT_META_KEYS = {"name", "output_dir"}


def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load a Composer-style experiment TOML file.

    The configuration should define an ``[experiment]`` section with
    metadata and a ``[pipelines.*]`` subsection for each pipeline. Shared
    defaults may be specified under ``[experiment.shared]`` and will be
    merged into every pipeline.

    The *path* argument names the TOML file to read; if *pipelines* is
    supplied only those sections are returned, otherwise every pipeline in
    the file is processed. The function returns a pair consisting of
    experiment metadata (including a merged ``"shared"`` dict) and a
    mapping from pipeline names to kwargs dictionaries that can be passed
    directly to :meth:`Composer.quick_run`.
    """
    try:
        with open(path, "rb") as f:
            cfg = toml.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e

    experiment_raw: Dict[str, Any] = cfg.get("experiment", {})
    shared: Dict[str, Any] = dict(experiment_raw.get("shared", {}))

    if "bounds" in shared and isinstance(shared["bounds"], list):
        shared["bounds"] = tuple(shared["bounds"])
    if "seeds" in shared and isinstance(shared["seeds"], list):
        shared["seeds"] = tuple(shared["seeds"])

    experiment_meta: Dict[str, Any] = {
        k: v for k, v in experiment_raw.items() if k in _EXPERIMENT_META_KEYS
    }
    experiment_meta["shared"] = shared

    raw_pipelines: Dict[str, Any] = cfg.get("pipelines", {})
    if not raw_pipelines:
        raise KeyError(f"No [pipelines.*] sections found in {path}")

    if pipelines is not None:
        missing = set(pipelines) - set(raw_pipelines)
        if missing:
            raise KeyError(
                f"Pipelines not found in {path}: {sorted(missing)}. "
                f"Available: {sorted(raw_pipelines.keys())}"
            )
        raw_pipelines = {k: raw_pipelines[k] for k in pipelines}

    resolved: Dict[str, Dict[str, Any]] = {}
    for name, pipeline_cfg in raw_pipelines.items():
        merged = {**shared, **pipeline_cfg}
        if "bounds" in merged and isinstance(merged["bounds"], list):
            merged["bounds"] = tuple(merged["bounds"])
        if "seeds" in merged and isinstance(merged["seeds"], list):
            merged["seeds"] = tuple(merged["seeds"])
        merged.setdefault("experiment_name", name)
        resolved[name] = merged

    return experiment_meta, resolved
