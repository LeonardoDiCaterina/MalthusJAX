"""
Fitness module for MalthusJAX.

This module provides NEW paradigm fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

# NEW architecture evaluators
from .base import BaseEvaluator, RegressionData
from .linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig, TENSORGP_FUNCTIONS, TENSORGP_NAMES
from .binary_evaluators import (
    BinarySumEvaluator, BinarySumConfig,
    KnapsackEvaluator, KnapsackConfig
)
from .real_evaluators import (
    SphereEvaluator, SphereConfig,
    GriewankEvaluator, GriewankConfig, 
    BoxEvaluator, BoxConfig
)

__all__ = [
    # Base classes
    "BaseEvaluator", "RegressionData",
    # NEW evaluators
    "LinearGPEvaluator", "LinearGPEvaluatorConfig", "OP_FUNCTIONS", "OP_NAMES",
    # Binary evaluators
    "BinarySumEvaluator", "BinarySumConfig",
    "KnapsackEvaluator", "KnapsackConfig", 
    # Real evaluators
    "SphereEvaluator", "SphereConfig",
    "GriewankEvaluator", "GriewankConfig",
    "BoxEvaluator", "BoxConfig",
]
