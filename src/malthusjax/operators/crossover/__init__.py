"""
Crossover operators for MalthusJAX.

Crossover operators generate offspring from parent pairs using the new batch-first paradigm.
All crossover operators inherit from BaseCrossover and return (num_offspring, genome_shape).
"""

from .binary import UniformCrossover, SinglePointCrossover
from .real import BlendCrossover, SimulatedBinaryCrossover, BinomialCrossover
from .linear import LinearCrossover

__all__ = [
    "UniformCrossover", 
    "SinglePointCrossover",
    "BlendCrossover",
    "SimulatedBinaryCrossover",
    "LinearCrossover",
]
