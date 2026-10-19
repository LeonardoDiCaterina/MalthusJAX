"""
Mutation Operators Module.
"""

from typing import Any

from .binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .categorical import ScrambleMutation as CategoricalScrambleMutation
from .categorical import SwapMutation as CategoricalSwapMutation

try:
    from .evosax_mutation import BatchedEvosaxGaussianWrapper, EvosaxGaussianWrapper

    _HAS_EVOSAX = True
except ImportError:
    _HAS_EVOSAX = False
    BatchedEvosaxGaussianWrapper = None  # type: ignore[assignment, misc]
    EvosaxGaussianWrapper = None  # type: ignore[assignment, misc]

from .real import (
    BallMutation,
    BallMutation_injection,
    BatchedGaussianMutation,
    GaussianMutation,
    GaussianMutation_injection,
    PolynomialMutation,
    PolynomialMutation_injection,
)

__all__ = [
    "BitFlipMutation",
    "ScrambleMutation",
    "SwapMutation",
    "CategoricalScrambleMutation",
    "CategoricalSwapMutation",
    "GaussianMutation",
    "GaussianMutation_injection",
    "BallMutation",
    "BallMutation_injection",
    "PolynomialMutation",
    "PolynomialMutation_injection",
    "BatchedGaussianMutation",
]
if _HAS_EVOSAX:
    __all__.extend(["EvosaxGaussianWrapper", "BatchedEvosaxGaussianWrapper"])

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def register_mutation_catalog(register_table_fn: Any) -> None:
    """Register mutation operators with the global catalog registry."""
    from typing import Callable, Dict, List, Tuple

    table: List[Tuple[str, Callable[..., Any], Dict[str, Any]]] = [
        # Real-valued mutation
        ("gaussian", GaussianMutation, {}),
        ("gaussian_injection", GaussianMutation_injection, {}),
        ("ball", BallMutation, {}),
        ("ball_injection", BallMutation_injection, {}),
        ("polynomial", PolynomialMutation, {}),
        ("polynomial_injection", PolynomialMutation_injection, {}),
        ("batched_gaussian", BatchedGaussianMutation, {}),
        # Binary mutation
        ("bitflip", BitFlipMutation, {}),
        ("scramble", ScrambleMutation, {}),
        ("swap", SwapMutation, {}),
        # Categorical mutation
        ("categorical_scramble", CategoricalScrambleMutation, {}),
        ("categorical_swap", CategoricalSwapMutation, {}),
    ]
    if _HAS_EVOSAX:
        table.extend(
            [
                ("evosax_gaussian", EvosaxGaussianWrapper, {}),
                ("batched_evosax_gaussian", BatchedEvosaxGaussianWrapper, {}),
            ]
        )
    register_table_fn(table, override=True)


def _register_mutation(register_table_fn: Any = None) -> None:
    """Backward-compatible registration helper."""
    if register_table_fn is None:
        from malthusjax.composer._registry import register_table

        register_table_fn = register_table
    register_mutation_catalog(register_table_fn)
