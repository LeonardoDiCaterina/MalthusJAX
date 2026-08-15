"""Simple TOML-based configuration loader for Composer experiments.

This lightweight utility exposes ``load_config`` and
``load_experiment_config`` for parsing pipeline and experiment definitions
without pulling in heavier dependencies; validation and schema handling are
performed at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml  # type: ignore[no-redef]

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


@dataclass
class ExperimentLoadResult:
    """Backward compatible config load result.

    Can be unpacked as `meta, pipelines = result` to support
    legacy code, or accessed directly via attributes for
    `data_registry`.
    """

    meta: Dict[str, Any]
    pipelines: Dict[str, Dict[str, Any]]
    data_registry: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Any:
        """Allow unpacking: meta, pipelines = result"""
        yield self.meta
        yield self.pipelines


def _parse_data_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract [data.*] sections from TOML."""
    data_registry: Dict[str, Any] = {}
    data_section = cfg.get("data", {})
    if not isinstance(data_section, dict):
        return data_registry

    for data_id, data_config in data_section.items():
        if isinstance(data_config, dict):
            data_registry[data_id] = data_config
    return data_registry


def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> ExperimentLoadResult:
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

    data_registry = _parse_data_section(cfg)

    return ExperimentLoadResult(
        meta=experiment_meta, pipelines=resolved, data_registry=data_registry
    )
