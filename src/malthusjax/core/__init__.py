"""
Core module for MalthusJAX.

This module contains the foundational Level 1 components:
- Base abstractions with JAX-native design
- Genome representations with automatic vectorization
- Fitness evaluation functions with symbiotic support
- Modern population management
"""

from . import base, fitness, genome
from .random import PRNGImpl, create_key

# Expose new base classes
from .base import BaseGenome, BasePopulation, DistanceMetric

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
