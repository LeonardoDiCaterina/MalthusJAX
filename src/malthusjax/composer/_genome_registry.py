"""Global genome registry for the GenomeCatalog.

Provides a lightweight ``register`` / ``register_table`` API that genome
modules call to register their specifications. ``GenomeCatalog`` reads the
accumulated entries via ``get_registry()``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

_GENOME_REGISTRY: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {}


def register(
    name: str,
    factory: Callable[..., Any],
    defaults: Dict[str, Any] | None = None,
    *,
    override: bool = False,
) -> None:
    """Register a single genome under *name*."""
    if not override and name in _GENOME_REGISTRY:
        raise KeyError(f"Genome '{name}' is already registered")
    _GENOME_REGISTRY[name] = (factory, defaults or {})


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
    return dict(_GENOME_REGISTRY)
