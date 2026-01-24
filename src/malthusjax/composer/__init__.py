"""Composer for building evolutionary experiments.

The Composer provides a product-first API for quickly running experiments
with sensible defaults and declarative configuration.
"""

try:
    from .composer import Composer
    __all__ = ["Composer"]
except ImportError:
    __all__ = []
