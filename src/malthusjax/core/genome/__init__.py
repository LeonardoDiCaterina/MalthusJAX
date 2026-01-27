"""
Genome module for MalthusJAX.

This module provides NEW paradigm genome implementations using @struct.dataclass
for evolutionary algorithms with JAX JIT compilation support.
"""

from .binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .categorical_genome import CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation
from .linear_genome import LinearGenome, LinearGenomeConfig, LinearPopulation
from .real_genome import RealGenome, RealGenomeConfig, RealPopulation

__all__ = [
    "LinearGenome",
    "LinearGenomeConfig",
    "LinearPopulation",
    "BinaryGenome",
    "BinaryGenomeConfig",
    "BinaryPopulation",
    "RealGenome",
    "RealGenomeConfig",
    "RealPopulation",
    "CategoricalGenome",
    "CategoricalGenomeConfig",
    "CategoricalPopulation",
]
