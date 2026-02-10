"""Global operator registry for the OperatorCatalog.

Provides a lightweight ``register`` / ``register_table`` API that operator
sub-packages call at import time. ``OperatorCatalog`` reads the accumulated
entries via ``get_registry()``, eliminating the need for a monolithic import
block and per-operator factory methods in *catalog.py*.

A registry entry maps a **catalog key** (e.g. ``"tournament"``) to a
**(factory, defaults)** pair.  The *factory* is any callable that accepts
``**kwargs`` and returns an operator instance.  *defaults* are merged
underneath user-provided kwargs so that ``catalog.get("tournament")``
still works with sensible values.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# catalog_key → (factory_callable, default_kwargs)
_OPERATOR_REGISTRY: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {}


def register(
    name: str,
    factory: Callable[..., Any],
    defaults: Dict[str, Any] | None = None,
    *,
    override: bool = False,
) -> None:
    """Register a single operator under *name*.

    Parameters
    ----------
    name
        Catalog key users will reference (e.g. ``"gaussian"``).
    factory
        Callable that accepts ``**kwargs`` and returns the operator.
    defaults
        Default kwargs applied when the user omits them in the spec string.
    override
        If ``False`` (default) and *name* is already registered, raise
        ``KeyError``.
    """
    if not override and name in _OPERATOR_REGISTRY:
        raise KeyError(f"Operator '{name}' is already registered")
    _OPERATOR_REGISTRY[name] = (factory, defaults or {})


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
    return dict(_OPERATOR_REGISTRY)
