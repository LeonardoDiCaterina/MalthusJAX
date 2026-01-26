"""
Fitness module for MalthusJAX.

This module provides NEW paradigm fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

# NEW architecture evaluators
from .base import BaseEvaluator, RegressionData
from .binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from .linear_gp_evaluator import (
    TENSORGP_FUNCTIONS,
    TENSORGP_NAMES,
    LinearGPEvaluator,
    LinearGPEvaluatorConfig,
)
from .real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "RegressionData",
    # NEW evaluators
    "LinearGPEvaluator",
    "LinearGPEvaluatorConfig",
    "TENSORGP_FUNCTIONS",
    "TENSORGP_NAMES",
    # Binary evaluators
    "BinarySumEvaluator",
    "BinarySumConfig",
    "KnapsackEvaluator",
    "KnapsackConfig",
    # Real evaluators
    "SphereEvaluator",
    "SphereConfig",
    "GriewankEvaluator",
    "GriewankConfig",
    "BoxEvaluator",
    "BoxConfig",
]
