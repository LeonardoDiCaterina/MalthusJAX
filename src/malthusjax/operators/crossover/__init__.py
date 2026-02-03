"""
Crossover operators for MalthusJAX.

Crossover operators generate offspring from parent pairs using the new batch-first paradigm.
All crossover operators inherit from BaseCrossover and return (num_offspring, genome_shape).
"""

from .binary import SinglePointCrossover, UniformCrossover
from .real import BinomialCrossover, BlendCrossover, SimulatedBinaryCrossover

__all__ = [
    "UniformCrossover",
    "SinglePointCrossover",
    "BlendCrossover",
    "BinomialCrossover",
    "SimulatedBinaryCrossover",
]
