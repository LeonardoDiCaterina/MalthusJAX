"""The Engine protocol — the one unified contract every backend satisfies.

This is the PyTorch nn.Module.forward() equivalent for MalthusJAX.
Every adapter, backend, and custom engine implements this protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Protocol, TypedDict, Union, runtime_checkable

import chex
import jax


@dataclass(frozen=True)
class EngineOutputOptions:
    """Per-execution controls shared by output-aware engine extensions."""

    capture_population_fingerprint: bool = False
    capture_solution: bool = True
    export_archive: bool = False
    collect_history: bool = True
    export_pareto: bool = False

    def __post_init__(self) -> None:
        for option in fields(self):
            if type(getattr(self, option.name)) is not bool:
                raise ValueError(f"{option.name} must be a boolean")


def population_fingerprint(genes: Any) -> Dict[str, str]:
    """Fingerprint ordered array leaves of the genomes actually used."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(genes)):
        digest.update(f"{leaf.dtype}:{leaf.shape}:".encode())
        digest.update(leaf.tobytes())
    return {
        "population_sha256": digest.hexdigest(),
        "population_hash_scheme": "jax-leaves-dtype-shape-bytes-v1",
    }


@runtime_checkable
class OutputConfigurableEngine(Protocol):
    def run_with_options(self, key: chex.Array, options: EngineOutputOptions) -> RunOutput:
        """Execute with explicitly requested outputs."""
        ...


class StepMetrics(TypedDict, total=False):
    """Per-generation metrics recorded in run history."""

    best_fitness: float
    mean_fitness: float
    std_fitness: float
    generation: int


class RunOutput(TypedDict, total=False):
    """Standard return type for Engine.run_once()."""

    history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timings: Dict[str, float]
    metric_metadata: Dict[str, Any]
    timing_metadata: Dict[str, Any]
    timing_record: Dict[str, Any]
    solution: Dict[str, Any]
    initialization: Dict[str, Any]
    archive: Dict[str, Any]
    pareto: Dict[str, Any]
    unavailable_outputs: Dict[str, str]
    unavailable_metrics: Dict[str, str]


@runtime_checkable
class Engine(Protocol):
    """The single protocol for all evolutionary engines.

    Any object with a run_once(key, ...) method is an Engine.
    This replaces implicit and duplicate protocol definitions.

    The protocol is intentionally minimal:
    - run_once() is the single required execution entry point.
    - Additional properties/methods are optional for introspection.
    """

    def run_once(self, key: chex.Array, **kwargs: Any) -> Union[RunOutput, Dict[str, Any]]:
        """Execute one complete evolutionary run and return results.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key for this run.
        \\*\\*kwargs : Any
            Optional execution parameters (e.g., step_logging).

        Returns
        -------
        RunOutput or Dict[str, Any]
            Dict with 'history', 'summary', and 'timings'.
        """
        ...
