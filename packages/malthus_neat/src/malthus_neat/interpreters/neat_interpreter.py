"""JAX-native neural network interpreter for NeatGenome.

Executes feed-forward or recurrent passes via pure matrix operations,
with full BaseInterpreter compliance, JIT compilation, and batch vmap support.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
from flax import struct

from malthus_neat.core.state import State
from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthusjax.core.fitness.composable.base import BaseInterpreter
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.interpreters.neat_interpreter")

_field: Any = struct.field


@struct.dataclass
class NeatInterpreter(BaseInterpreter[NeatGenome]):  # type: ignore[misc]
    """Interprets and executes a NeatGenome on input feature arrays.

    Attributes:
        genome_config: Structural specification for inputs, outputs, and capacity.
    """

    genome_config: NeatGenomeConfig = _field(pytree_node=False, default_factory=NeatGenomeConfig)
    _tn_genome: Any = _field(pytree_node=False, default=None)
    _state: Any = _field(pytree_node=False, default=None)

    def __post_init__(self) -> None:
        if self._tn_genome is None:
            object.__setattr__(self, "_tn_genome", self.genome_config.to_tensorneat_genome())
        if self._state is None:
            st = State().register(randkey=jax.random.PRNGKey(0))
            st = self._tn_genome.setup(st)
            object.__setattr__(self, "_state", st)

    @property
    def num_params(self) -> int:
        """Dynamic parameter count representation."""
        return self.genome_config.max_conns

    def apply(self, genome: NeatGenome, inputs: chex.Array) -> chex.Array:
        """Executes the network on a single 1D input feature vector or 2D batch of samples."""
        if inputs.ndim > 1:
            return self.evaluate_batch(genome, inputs)
        transformed = self._tn_genome.transform(self._state, genome.nodes, genome.conns)
        return self._tn_genome.forward(self._state, transformed, inputs)

    def evaluate_batch(self, genome: NeatGenome, X: chex.Array) -> chex.Array:
        """Evaluates a single genome on a batch of samples (N, D) using jax.vmap."""
        transformed = self._tn_genome.transform(self._state, genome.nodes, genome.conns)
        return jax.vmap(self._tn_genome.forward, in_axes=(None, None, 0))(
            self._state, transformed, X
        )

    def evaluate_population(self, population_genes: NeatGenome, X: chex.Array) -> chex.Array:
        """Vectorized execution across both population batch and sample batch.

        Args:
            population_genes: Batched NeatGenome with nodes shape (B, N, 5) and conns (B, C, 3).
            X: Input feature array with shape (N_samples, D).

        Returns:
            Predictions of shape (B, N_samples, num_outputs) or (B, N_samples).
        """
        transformed_pop = jax.vmap(self._tn_genome.transform, in_axes=(None, 0, 0))(
            self._state, population_genes.nodes, population_genes.conns
        )

        def _forward_one_genome(params: Any) -> chex.Array:
            return jax.vmap(self._tn_genome.forward, in_axes=(None, None, 0))(
                self._state, params, X
            )

        return jax.vmap(_forward_one_genome)(transformed_pop)


__all__ = ["NeatInterpreter"]
