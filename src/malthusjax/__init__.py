"""
MalthusJAX: High-Performance Evolutionary Computation in JAX.
"""

__version__ = "0.2.0"

# 1. Stabilize runtime environment and hook native crash diagnostics
from .core.diagnostics import (
    format_crash_banner,
    get_environment_diagnostics,
    install_crash_handler,
    print_environment_diagnostics,
    stabilize_runtime_environment,
    uninstall_crash_handler,
)

stabilize_runtime_environment()
install_crash_handler()

import logging

# Attach NullHandler to root library logger according to PEP 282
logging.getLogger("malthusjax").addHandler(logging.NullHandler())

from .core.base import BaseGenome, BasePopulation, DistanceMetric
from .core.fitness.base import BaseEvaluator
from .core.fitness.binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from .core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
    CategoricalPopulation,
)
from .core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from .core.logger import (
    StepLoggingConfig,
    configure_logging,
    get_logger,
    set_log_level,
)
from .operators import crossover, mutation, selection

# Explicitly define what `from malthusjax import *` exports

__all__ = [
    # Submodules
    "selection",
    "crossover",
    "mutation",
    # Diagnostics & stabilization
    "stabilize_runtime_environment",
    "install_crash_handler",
    "uninstall_crash_handler",
    "format_crash_banner",
    "get_environment_diagnostics",
    "print_environment_diagnostics",
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
    # Logging subsystem
    "get_logger",
    "set_log_level",
    "configure_logging",
    "StepLoggingConfig",
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
