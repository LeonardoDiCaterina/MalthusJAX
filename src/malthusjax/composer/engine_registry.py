"""Global engine registry for the EngineRegistry catalog.

Provides a lightweight ``register`` / ``register_table`` API that engine
modules call at import time.  ``EngineRegistry`` (the catalog class in
*engine_catalog.py*) reads the accumulated entries via ``get_registry()``.

A registry entry maps a **catalog key** (e.g. ``"ga"``) to a
**(factory, defaults)** pair.  The *factory* is any callable that accepts
operator instances + ``**kwargs`` and returns an engine adapter compatible
with the :class:`~malthusjax.benchmarking.runner.Engine` protocol.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

_ENGINE_REGISTRY: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {}


def register(
    name: str,
    factory: Callable[..., Any],
    defaults: Dict[str, Any] | None = None,
    *,
    override: bool = False,
) -> None:
    """Register a single engine under *name*.

    Parameters
    ----------
    name
        Engine key users will reference (e.g. ``"ga"``, ``"nsga2"``).
    factory
        Callable that accepts ``(evaluator, selection, crossover, mutation,
        **kwargs)`` and returns an engine satisfying the
        :class:`~malthusjax.benchmarking.runner.Engine` protocol.
    defaults
        Default kwargs applied when the user omits them in the spec string.
    override
        If ``False`` (default) and *name* is already registered, raise
        ``KeyError``.
    """
    if not override and name in _ENGINE_REGISTRY:
        raise KeyError(f"Engine '{name}' is already registered")
    _ENGINE_REGISTRY[name] = (factory, defaults or {})


def register_table(
    entries: list[Tuple[str, Callable[..., Any], Dict[str, Any]]],
    *,
    override: bool = False,
) -> None:
    """Bulk-register a list of ``(name, factory, defaults)`` tuples."""
    for name, factory, defaults in entries:
        register(name, factory, defaults, override=override)


def get_registry() -> Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]]:
    """Return a **copy** of the current registry."""
    return dict(_ENGINE_REGISTRY)


def list_available() -> List[str]:
    """Return sorted list of registered engine names."""
    return sorted(_ENGINE_REGISTRY.keys())
