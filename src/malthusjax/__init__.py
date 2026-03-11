"""
MalthusJAX: High-Performance Evolutionary Computation in JAX.
"""

__version__ = "0.2.0"

from .core.base import BaseGenome, BasePopulation, DistanceMetric
from .core.fitness.base import BaseEvaluator
from .core.fitness.binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from .core.fitness.real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)
from .core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
    CategoricalPopulation,
)
from .core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from .operators import crossover, mutation, selection

# Explicitly define what `from malthusjax import *` exports
__all__ = [
    # Submodules
    "selection",
    "crossover",
    "mutation",
    # Core top-level types
    "BaseGenome",
    "BasePopulation",
    "DistanceMetric",
    # Evaluator bases & configs
    "BaseEvaluator",
    "BinarySumConfig",
    "BinarySumEvaluator",
    "KnapsackConfig",
    "KnapsackEvaluator",
    "BoxConfig",
    "BoxEvaluator",
    "GriewankConfig",
    "GriewankEvaluator",
    "SphereConfig",
    "SphereEvaluator",
    # Genomes & populations
    "BinaryGenome",
    "BinaryGenomeConfig",
    "BinaryPopulation",
    "CategoricalGenome",
    "CategoricalGenomeConfig",
    "CategoricalPopulation",
    "RealGenome",
    "RealGenomeConfig",
    "RealPopulation",
    # Engine abstractions & helpers
    "AbstractEngine",
    "AbstractEngineParams",
    "AbstractEvolutionState",
    "compute_unroll_num",
    "GeneticEngine",
    "GeneticEngineParams",
    "GeneticGenerationOutput",
    "ScheduleType",
    "compute_scheduled_strength",
]

# --- 3. ENGINE (Top Level) ---
from .engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    compute_unroll_num,
)
from .engine.genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
from .engine.schedules import ScheduleType, compute_scheduled_strength
# from .engine.diversity_engine import DiversityAwareEngine
