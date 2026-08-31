"""LSP NeuralGenome — LinearGenome extended with continuous per-connection weights.

Implements the CGPANN / MEP-NN node equation (Phase 2 of the architecture roadmap):

    N_i = activation( sum_k( weights[i,k] * memory[args[i,k]] ) )

where memory stores vectors of shape (output_dim,), so each row produces an
(output_dim,) output.  This allows the genome to directly parameterise a
variable-topology neural network whose weights are evolved alongside the
structure.
"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.linear import BasePrefixAwareGenome, PrefixGenomeConfig

# ---------------------------------------------------------------------------
# Activations catalogue
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, Callable[[chex.Array], chex.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda x: x,
}


# ---------------------------------------------------------------------------
# NeuralGenomeConfig
# ---------------------------------------------------------------------------


@struct.dataclass
class NeuralGenomeConfig(PrefixGenomeConfig):
    """Configuration for NeuralGenome.

    Extends ``PrefixGenomeConfig`` with:

    Attributes:
        activation: Name of the activation function (key in ``ACTIVATIONS``).
        output_dim: Dimensionality of each node's output vector.
    """

    activation: str = struct.field(pytree_node=False, default="tanh")
    output_dim: int = struct.field(pytree_node=False, default=1)


# ---------------------------------------------------------------------------
# NeuralGenome
# ---------------------------------------------------------------------------


@struct.dataclass
class NeuralGenome(BasePrefixAwareGenome):
    """LinearGenome + continuous per-connection weights.

    Fields:
        ops:     (L,)            int32   — opcode indices (for compatibility;
                                           currently only one op used: neuron).
        args:    (L, max_arity)  int32   — argument pointer indices.
        weights: (L, max_arity)  float32 — scalar weight for each connection.

    The node computation is::

        out[i] = activation( sum_k weights[i,k] * memory[args[i,k]] )

    where ``memory`` is a ``(N + L, output_dim)`` buffer — input slots hold
    the observation broadcast to ``output_dim`` dimensions, intermediate slots
    accumulate the row outputs.
    """

    weights: chex.Array  # (L, max_arity), float32

    @classmethod
    def random_init(  # type: ignore[override]
        cls,
        key: chex.PRNGKey,
        config: NeuralGenomeConfig,  # type: ignore[override]
    ) -> "NeuralGenome":
        """Randomly initialise topology (ops, args) and weights (Glorot)."""
        k_struct, k_weights = jax.random.split(key)
        base = BasePrefixAwareGenome.random_init(k_struct, config)

        fan_in = config.max_arity
        std = jnp.sqrt(2.0 / fan_in)
        weights = jax.random.normal(k_weights, (config.length, config.max_arity)) * std

        return cls(ops=base.ops, args=base.args, weights=weights)


__all__ = ["NeuralGenomeConfig", "NeuralGenome", "ACTIVATIONS"]
