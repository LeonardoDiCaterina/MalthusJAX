"""
Mutation Operators Module.
"""

from .binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .categorical import ScrambleMutation as CategoricalScrambleMutation
from .categorical import SwapMutation as CategoricalSwapMutation
from .evosax_mutation import BatchedEvosaxGaussianWrapper, EvosaxGaussianWrapper
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
    "EvosaxGaussianWrapper",
    "BatchedGaussianMutation",
    "BatchedEvosaxGaussianWrapper",
]

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def _register_mutation() -> None:
    """Register mutation operators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            # Real-valued mutation
            ("gaussian", GaussianMutation, {}),
            ("gaussian_injection", GaussianMutation_injection, {}),
            ("ball", BallMutation, {}),
            ("ball_injection", BallMutation_injection, {}),
            ("polynomial", PolynomialMutation, {}),
            ("polynomial_injection", PolynomialMutation_injection, {}),
            ("evosax_gaussian", EvosaxGaussianWrapper, {}),
            ("batched_gaussian", BatchedGaussianMutation, {}),
            ("batched_evosax_gaussian", BatchedEvosaxGaussianWrapper, {}),
            # Binary mutation
            ("bitflip", BitFlipMutation, {}),
            ("scramble", ScrambleMutation, {}),
            ("swap", SwapMutation, {}),
            # Categorical mutation
            ("categorical_scramble", CategoricalScrambleMutation, {}),
            ("categorical_swap", CategoricalSwapMutation, {}),
        ],
        override=True,
    )


_register_mutation()
