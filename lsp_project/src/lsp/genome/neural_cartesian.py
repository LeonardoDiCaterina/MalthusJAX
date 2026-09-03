"""LSP Neural Cartesian Genome.

Implements the dCGPANN node equation:
    N_i = activation( sum_k( weights[i,k] * N_{args[i,k]} ) + biases[i] )

Extends the standard CartesianGenome with continuous parameters (weights and biases).
"""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig

# ---------------------------------------------------------------------------
# Activations catalogue
# ---------------------------------------------------------------------------

ACTIVATIONS: dict[str, Callable[[chex.Array], chex.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "linear": lambda x: x,
    "elu": jax.nn.elu,
}

ACTIVATIONS_LIST = list(ACTIVATIONS.values())


from malthusjax.composer.decorators import register_genome

@register_genome(
    name="neural_cartesian",
    defaults={
        "num_rows": 1,
        "num_cols": 10,
        "levels_back": 10,
        "num_inputs": 2,
        "num_outputs": 1,
        "num_ops": 5,
        "max_arity": 2,
    }
)
@struct.dataclass
class NeuralCartesianGenomeConfig(CartesianGenomeConfig):
    """Configuration for NeuralCartesianGenome.

    Inherits topology constraints from CartesianGenomeConfig, but configures
    ops to be indices into the ACTIVATIONS list.
    """

    def init_population(self, key: chex.PRNGKey, size: int) -> "NeuralCartesianPopulation":
        from lsp.genome.neural_cartesian import NeuralCartesianPopulation
        return NeuralCartesianPopulation.init_random(key, self, size)

    @property
    def dtype(self) -> Any:
        # Override to float32 so the engine's _enforce_layout doesn't cast
        # our continuous weights/biases to int32!
        return jnp.float32


from malthusjax.composer.decorators import register_genome

@struct.dataclass
class NeuralCartesianGenome(CartesianGenome):
    """CartesianGenome + continuous parameters (weights and biases).

    Fields:
        ops:     (num_nodes,)            int32   - index into ACTIVATIONS list
        args:    (num_nodes, max_arity)  int32   - connection pointers
        out_nodes: (num_outputs,)        int32   - pointers to final output nodes
        weights: (num_nodes, max_arity)  float32 - connection weights
        biases:  (num_nodes,)            float32 - node biases
    """

    weights: chex.Array
    biases: chex.Array

    @classmethod
    def random_init(
        cls,
        key: chex.PRNGKey,
        config: NeuralCartesianGenomeConfig,
    ) -> "NeuralCartesianGenome":
        """Randomly initialize topology and continuous weights."""
        k_struct, k_weights, k_biases = jax.random.split(key, 3)

        # Initialize standard Cartesian topology
        base = CartesianGenome.random_init(k_struct, config)

        # Initialize continuous parameters
        # Glorot Normal for weights
        fan_in = config.max_arity
        std = jnp.sqrt(2.0 / fan_in)
        weights = jax.random.normal(k_weights, (config.num_nodes, config.max_arity)) * std

        # Zeros for biases
        biases = jnp.zeros((config.num_nodes,))

        return cls(
            ops=base.ops,
            args=base.args,
            out_nodes=base.out_nodes,
            weights=weights,
            biases=biases
        )

@struct.dataclass
class NeuralCartesianPopulation(BasePopulation[NeuralCartesianGenome]):
    genes: NeuralCartesianGenome
    fitness: chex.Array
    config: NeuralCartesianGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: NeuralCartesianGenomeConfig, size: int) -> "NeuralCartesianPopulation":
        keys = jax.random.split(key, size)
        batched_genes = jax.vmap(NeuralCartesianGenome.random_init, in_axes=(0, None))(keys, config)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)

__all__ = ["NeuralCartesianGenomeConfig", "NeuralCartesianGenome", "ACTIVATIONS", "ACTIVATIONS_LIST", "NeuralCartesianPopulation"]
