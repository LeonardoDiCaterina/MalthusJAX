"""
Genetic operators for evolutionary algorithms.

Provides mutation, crossover, and selection operators organized by type.
All operators follow the NEW @struct.dataclass paradigm with factory methods.
"""

# Base operator abstractions
from .base import BaseMutation, BaseCrossover, BaseSelection

# Crossover operators  
from .crossover.binary import UniformCrossover, SinglePointCrossover
from .crossover.real import BlendCrossover, SimulatedBinaryCrossover, SimulatedBinaryCrossover,  UniformCrossover as realUniform
from .crossover.linear import LinearCrossover

# Mutation operators
from .mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .mutation.real import GaussianMutation, BallMutation, PolynomialMutation
from .mutation.categorical import CategoricalFlipMutation, RandomCategoryMutation
from .mutation.linear import LinearMutation, LinearPointMutation

# Selection operators
from .selection.tournament import TournamentSelection
from .selection.roulette import RouletteSelection
from .selection.elite_pool import ElitePoolSelection

__all__ = [
    # Base abstractions
    "BaseMutation", "BaseCrossover", "BaseSelection",
    # Crossover operators
    "UniformCrossover", "SinglePointCrossover", 
    "BlendCrossover", "SimulatedBinaryCrossover",
    "LinearCrossover", "realUniform",
    # Mutation operators
    "BitFlipMutation", "ScrambleMutation", "SwapMutation",
    "GaussianMutation", "BallMutation", "PolynomialMutation", 
    "CategoricalFlipMutation", "RandomCategoryMutation",
    "LinearMutation", "LinearPointMutation",
    # Selection operators
    "TournamentSelection", "RouletteSelection", "ElitePoolSelection"
]