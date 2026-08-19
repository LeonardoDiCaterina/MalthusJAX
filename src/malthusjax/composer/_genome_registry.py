"""Global genome registry for the GenomeCatalog.

Provides a lightweight ``register`` / ``register_table`` API that genome
modules call to register their specifications. ``GenomeCatalog`` reads the
accumulated entries via ``get_registry()``.
"""

from __future__ import annotations

from ._shared_registry import make_catalog_registry

register, register_table, get_registry, list_available, _GENOME_REGISTRY = make_catalog_registry("Genome")

register.__doc__ = """Register a single genome under *name*."""

register_table.__doc__ = """Bulk-register a list of ``(name, factory, defaults)`` tuples."""
