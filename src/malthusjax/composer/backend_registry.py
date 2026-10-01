"""Backend registry — maps backend name strings to BackendProvider instances.

Built on top of the shared ``make_catalog_registry`` infrastructure used by
OperatorCatalog, EngineRegistry, and GenomeCatalog.

Public API::

    register_backend("evosax", EvosaxProvider())
    get_backends()["evosax"]   # -> (provider, defaults, metadata)
    list_backends()            # -> ["evosax", "malthusjax", ...]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._shared_registry import make_catalog_registry

_raw_register_backend, register_backend_table, get_backends, list_backends, _BACKEND_REGISTRY = (
    make_catalog_registry("Backend")
)


def register_backend(
    name: str,
    provider: Any,
    defaults: Optional[Dict[str, Any]] = None,
    override: bool = False,
    **kwargs: Any,
) -> None:
    """Register a backend provider with protocol validation.

    Parameters
    ----------
    name : str
        Canonical backend identifier.
    provider : BackendProvider
        Provider instance conforming to BackendProvider protocol.
    defaults : Optional[Dict[str, Any]]
        Optional default parameters attached to this provider.
    override : bool
        Whether to overwrite an existing registration.

    Raises
    ------
    TypeError
        If provider does not implement the required BackendProvider interface methods.
    """

    missing = []
    for method in ("build_engine", "resolve_evaluator", "default_strategy"):
        if not hasattr(provider, method) or not callable(getattr(provider, method, None)):
            missing.append(method)
    if missing:
        raise TypeError(
            f"Backend '{name}' does not satisfy BackendProvider protocol. "
            f"Missing or non-callable methods: {missing}"
        )
    _raw_register_backend(name, provider, defaults=defaults, override=override, **kwargs)
