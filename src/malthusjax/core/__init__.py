"""
Core module for MalthusJAX.

This module contains the foundational Level 1 components:
- Base abstractions with JAX-native design
- Genome representations with automatic vectorization
- Fitness evaluation functions with symbiotic support
- Modern population management
"""

from . import base, fitness, genome

# Expose new base classes
from .base import BaseGenome, BasePopulation, DistanceMetric
from .random import PRNGImpl, create_key

__all__ = [
    "base",
    "genome",
    "fitness",
    "BaseGenome",
    "BasePopulation",
    "DistanceMetric",
    "PRNGImpl",
    "create_key",
]
