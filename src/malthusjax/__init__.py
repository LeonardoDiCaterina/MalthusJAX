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

# --- 2. OPERATOR NAMESPACES (Grouped) ---

class mutation:
    """Namespace for Mutation Operators."""
    # Binary
    from .operators.mutation.binary import BitFlipMutation as BitFlip
    from .operators.mutation.binary import ScrambleMutation as Scramble
    from .operators.mutation.binary import SwapMutation as Swap  
    
    # Real
    from .operators.mutation.real import GaussianMutation as Gaussian
    from .operators.mutation.real import BallMutation as Ball
    from .operators.mutation.real import PolynomialMutation as Polynomial
    
    # Categorical
    from .operators.mutation.categorical import CategoricalFlipMutation as CategoryFlip
    from .operators.mutation.categorical import RandomCategoryMutation as RandomCategory
    
    # Linear
    from .operators.mutation.linear import LinearMutation as Linear
    from .operators.mutation.linear import LinearPointMutation as LinearPoint

class crossover:
    """Namespace for Crossover Operators."""
    # Binary
    from .operators.crossover.binary import UniformCrossover as Uniform
    from .operators.crossover.binary import SinglePointCrossover as SinglePoint
    
    # Real
    from .operators.crossover.real import BlendCrossover as Blend
    from .operators.crossover.real import SimulatedBinaryCrossover as SBX
    
    # Linear
    from .operators.crossover.linear import LinearCrossover as Linear

class selection:
    """Namespace for Selection Operators."""
    from .operators.selection.tournament import TournamentSelection as Tournament
    from .operators.selection.roulette import RouletteWheelSelection as Roulette

# --- 3. ENGINE (Top Level) ---
from .engine.base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams
from .engine.genetic_engine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput