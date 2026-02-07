"""
Genetic operators for evolutionary algorithms.

Provides mutation, crossover, and selection operators organized by type.
All operators follow the NEW @struct.dataclass paradigm with factory methods.
"""

# Base operator abstractions
from .base import BaseCrossover, BaseMutation, BaseSelection
from .base_injection import BaseCrossover_injection, BaseMutation_injection

# Crossover operators
from .crossover.binary import SinglePointCrossover, UniformCrossover
from .crossover.real import BlendCrossover, SimulatedBinaryCrossover
from .crossover.real import UniformCrossover as realUniform

# Mutation operators
from .mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .mutation.real import BallMutation, GaussianMutation, PolynomialMutation
from .selection.elite_pool import ElitePoolSelection
from .selection.roulette import RouletteSelection

# Selection operators
from .selection.tournament import TournamentSelection

__all__ = [
    # Base abstractions
    "BaseMutation",
    "BaseCrossover",
    "BaseSelection",
    "BaseMutation_injection",
    "BaseCrossover_injection",
    # Crossover operators
    "UniformCrossover",
    "SinglePointCrossover",
    "BlendCrossover",
    "SimulatedBinaryCrossover",
    "realUniform",
    # Mutation operators
    "BitFlipMutation",
    "ScrambleMutation",
    "SwapMutation",
    "GaussianMutation",
    "BallMutation",
    "PolynomialMutation",
    # Selection operators
    "TournamentSelection",
    "RouletteSelection",
    "ElitePoolSelection",
]
