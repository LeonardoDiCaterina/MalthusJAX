"""Composer for building evolutionary experiments.

The Composer provides a product-first API for quickly running experiments
with sensible defaults and declarative configuration.
"""

from .composer import Composer
from .decorators import (
    register_crossover,
    register_engine,
    register_fitness,
    register_genome,
    register_mutation,
    register_selection,
)
from .engine_catalog import EngineRegistry
from .evosax_adapter import EvosaxEngineAdapter, build_evosax_engine, list_strategies
from .mo_factory import MOEngineAdapter, build_mo_engine

__all__ = [
    "Composer",
    "EngineRegistry",
    "EvosaxEngineAdapter",
    "build_evosax_engine",
    "list_strategies",
    "register_selection",
    "register_mutation",
    "register_crossover",
    "register_fitness",
    "register_engine",
    "register_genome",
    "MOEngineAdapter",
    "build_mo_engine",
]
