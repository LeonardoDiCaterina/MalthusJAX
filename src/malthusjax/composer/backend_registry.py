"""Backend registry — maps backend name strings to BackendProvider instances.

Built on top of the shared ``make_catalog_registry`` infrastructure used by
OperatorCatalog, EngineRegistry, and GenomeCatalog.

Public API::

    register_backend("evosax", EvosaxProvider())
    get_backends()["evosax"]   # -> (provider, defaults, metadata)
    list_backends()            # -> ["evosax", "malthusjax", ...]
"""

from __future__ import annotations

from ._shared_registry import make_catalog_registry

register_backend, register_backend_table, get_backends, list_backends, _BACKEND_REGISTRY = (
    make_catalog_registry("Backend")
)
