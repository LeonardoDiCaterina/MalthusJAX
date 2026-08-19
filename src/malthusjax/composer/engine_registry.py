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

from ._shared_registry import make_catalog_registry

register, register_table, get_registry, list_available, _ENGINE_REGISTRY = make_catalog_registry("Engine")

register.__doc__ = """Register a single engine under *name*.

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

register_table.__doc__ = """Bulk-register a list of ``(name, factory, defaults)`` tuples."""
