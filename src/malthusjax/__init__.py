"""
MalthusJAX: High-Performance Evolutionary Computation in JAX.
"""

__version__ = "0.2.0"

# --- 1. CORE COMPONENTS (Top Level) ---
from .core.base import BaseGenome, BasePopulation, DistanceMetric
from .core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from .core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation
from .core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation

# Evaluators
from .core.fitness.base import BaseEvaluator
from .core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig, KnapsackEvaluator, KnapsackConfig
from .core.fitness.real_evaluators import SphereEvaluator, SphereConfig, GriewankEvaluator, GriewankConfig, BoxEvaluator, BoxConfig
from .core.fitness.linear_gp_evaluator import LinearGPEvaluator

# --- 2. OPERATORS (Top Level + Namespaces) ---
# Import operators at top level for direct access: mjx.TournamentSelection()
from .operators.selection.tournament import TournamentSelection
from .operators.selection.roulette import RouletteWheelSelection
from .operators.crossover.binary import UniformCrossover, SinglePointCrossover
from .operators.crossover.real import BlendCrossover, SimulatedBinaryCrossover
from .operators.crossover.linear import LinearCrossover
from .operators.mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .operators.mutation.real import GaussianMutation, BallMutation, PolynomialMutation
from .operators.mutation.categorical import CategoricalFlipMutation, RandomCategoryMutation
from .operators.mutation.linear import LinearMutation, LinearPointMutation

# Also provide namespaces for organized access: mjx.selection.Tournament()
class selection:
    """Namespace for Selection Operators."""
    from .operators.selection.tournament import TournamentSelection as Tournament
    from .operators.selection.roulette import RouletteWheelSelection as Roulette

class crossover:
    """Namespace for Crossover Operators."""
    from .operators.crossover.binary import UniformCrossover as Uniform
    from .operators.crossover.binary import SinglePointCrossover as SinglePoint
    from .operators.crossover.real import BlendCrossover as Blend
    from .operators.crossover.real import SimulatedBinaryCrossover as SBX
    from .operators.crossover.linear import LinearCrossover as Linear

class mutation:
    """Namespace for Mutation Operators."""
    from .operators.mutation.binary import BitFlipMutation as BitFlip
    from .operators.mutation.binary import ScrambleMutation as Scramble
    from .operators.mutation.binary import SwapMutation as Swap
    from .operators.mutation.real import GaussianMutation as Gaussian
    from .operators.mutation.real import BallMutation as Ball
    from .operators.mutation.real import PolynomialMutation as Polynomial
    from .operators.mutation.categorical import CategoricalFlipMutation as CategoryFlip
    from .operators.mutation.categorical import RandomCategoryMutation as RandomCategory
    from .operators.mutation.linear import LinearMutation as Linear
    from .operators.mutation.linear import LinearPointMutation as LinearPoint

# --- 3. ENGINE (Top Level) ---
from .engine.base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams
from .engine.genetic_engine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
from .engine.diversity_engine import DiversityAwareEngine