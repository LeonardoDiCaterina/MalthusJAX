"""Composer for building evolutionary experiments.

The Composer provides a product-first API for quickly running experiments
with sensible defaults and declarative configuration.
"""

from .composer import Composer
from .evosax_adapter import EvosaxEngineAdapter, build_evosax_engine, list_strategies

__all__ = ["Composer", "EvosaxEngineAdapter", "build_evosax_engine", "list_strategies"]
