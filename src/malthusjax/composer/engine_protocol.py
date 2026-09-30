"""The Engine protocol — the one unified contract every backend satisfies.

This is the PyTorch nn.Module.forward() equivalent for MalthusJAX.
Every adapter, backend, and custom engine implements this protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, TypedDict, Union, runtime_checkable

import chex


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
        **kwargs : Any
            Optional execution parameters (e.g., step_logging).

        Returns
        -------
        RunOutput or Dict[str, Any]
            Dict with 'history', 'summary', and 'timings'.
        """
        ...
