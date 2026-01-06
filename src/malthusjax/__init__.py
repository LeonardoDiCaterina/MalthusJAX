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


# --- 2. OPERATORS ---
# Import the submodules entirely. This enables usage like: mjx.selection.TournamentSelection
from .operators import selection
from .operators import crossover
from .operators import mutation

# --- 4. EXPORTS ---
# Explicitly define what `from malthusjax import *` exports
__all__ = [
    # Submodules
    "selection",
    "crossover",
    "mutation",
    # Core Genomes
    "BaseGenome", "BasePopulation",
    "BinaryGenome", "BinaryGenomeConfig",
    "RealGenome", "RealGenomeConfig",
    "CategoricalGenome", "CategoricalGenomeConfig",
    "LinearGenome", "LinearGenomeConfig",
    # Engines
    "GeneticEngine", "GeneticEngineParams", "GeneticGenerationOutput",
]

# --- 3. ENGINE (Top Level) ---
from .engine.base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams
from .engine.genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
#from .engine.diversity_engine import DiversityAwareEngine