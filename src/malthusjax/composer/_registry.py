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

from ._shared_registry import make_catalog_registry

register, register_table, get_registry, list_available, _OPERATOR_REGISTRY = make_catalog_registry("Operator")

# Keep the original docstrings for the exported functions
register.__doc__ = """Register a single operator under *name*.

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

register_table.__doc__ = """Bulk-register a list of ``(name, factory, defaults)`` tuples."""
