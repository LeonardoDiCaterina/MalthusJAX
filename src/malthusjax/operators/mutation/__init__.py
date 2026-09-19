"""
Mutation Operators Module.
"""

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


def _register_mutation() -> None:
    """Register mutation operators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    table = [
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
    register_table(table, override=True)


_register_mutation()
