"""Concrete Linear GP prefix evaluator for symbolic regression.

Implements :meth:`evaluate_all_prefixes` by executing the genome
interpreter and computing per-row MSE against regression targets.
Uses the standalone ``predict_one`` function (Option B).
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig, RegressionData
from malthusjax.core.fitness.linear_gp_interpreter import predict_one
from malthusjax.core.fitness.prefix.evaluator import BasePrefixEvaluator
from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome


@struct.dataclass
class LinearGPPrefixEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for the Linear GP Prefix Evaluator.

    Attributes:
        num_inputs: Number of external input variables (N).
        length: Number of instructions / rows (L).
    """

    num_inputs: int = struct.field(pytree_node=False, default=10)  # type: ignore[no-untyped-call]
    length: int = struct.field(pytree_node=False, default=100)  # type: ignore[no-untyped-call]


@struct.dataclass
class LinearGPPrefixEvaluator(
    BasePrefixEvaluator[LinearGPPrefixEvaluatorConfig, RegressionData]
):
    """Linear GP evaluator returning per-row MSE for symbolic regression.

    Instead of reducing to a single scalar (as :class:`LinearGPEvaluator`
    does), this evaluator returns the full ``(L,)`` MSE vector so that
    the :class:`FlatPoolEngine` can operate on every prefix independently.

    Uses the standalone :func:`predict_one` interpreter (Option B) to
    avoid code duplication with the original evaluator.
    """

    def evaluate_all_prefixes(
        self, genome: BasePrefixAwareGenome
    ) -> chex.Array:
        """Compute MSE for every instruction row.

        Args:
            genome: An unbatched ``BasePrefixAwareGenome``.

        Returns:
            Array of shape ``(L,)`` where entry *l* is the MSE of the
            program that reads out from instruction row *l*.
        """
        from functools import partial

        X, y = self.data

        # Bind the static config args; vmap maps over x_input only
        _interp = partial(
            predict_one,
            num_inputs=self.config.num_inputs,
            length=self.config.length,
        )
        # Execute the genome on every data point → (n_samples, L)
        all_preds = jax.vmap(_interp, in_axes=(None, 0))(genome, X)

        # Per-row MSE: (n_samples, L) → (L,)
        squared_errors = jnp.square(all_preds - y[:, None])
        mse_per_row = jnp.mean(squared_errors, axis=0)

        return mse_per_row

    def get_program_prediction(
        self,
        genome: BasePrefixAwareGenome,
        X: chex.Array,
        instruction_idx: int = -1,
    ) -> chex.Array:
        """Retrieve data-wide predictions from a specific instruction row.

        Args:
            genome: An unbatched genome.
            X: Input data array of shape ``(n_samples, num_inputs)``.
            instruction_idx: Which row to read out from (default: last).

        Returns:
            Predictions of shape ``(n_samples,)``.
        """
        from functools import partial

        _interp = partial(
            predict_one,
            num_inputs=self.config.num_inputs,
            length=self.config.length,
        )
        all_outputs = jax.vmap(_interp, in_axes=(None, 0))(genome, X)
        return all_outputs[:, instruction_idx]
