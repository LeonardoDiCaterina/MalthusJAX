"""Differentiable Cartesian Genetic Programming (dCGP / dCGPANN) Genome.

Combines the discrete DAG topology of Cartesian GP with continuous edge weights
and node biases, enabling high-order AutoDiff and gradient-based memetic search
(Izzo et al., 2016; Märtens & Izzo, 2019).
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.cartesian_genome import (
    CartesianGenome,
    CartesianGenomeConfig,
    CartesianPopulation,
)
from malthusjax.composer import register_genome
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.genomes.differentiable_cartesian_genome")

_field: Any = struct.field


@register_genome("diff_cartesian_genome", override=True)
@struct.dataclass
class DifferentiableCartesianGenomeConfig(CartesianGenomeConfig):
    """Configuration for Differentiable Cartesian GP genomes.

    Attributes:
        weight_init_std: Scaling factor for Glorot normal weight initialization.
    """

    weight_init_std: float = _field(pytree_node=False, default=1.0)

    @property
    def dtype(self) -> Any:
        # Override to float32 so engine does not cast continuous weights/biases to int32
        return jnp.float32

    def init_population(self, key: chex.PRNGKey, size: int) -> "DifferentiableCartesianPopulation":
        return DifferentiableCartesianPopulation.init(key, self, size)


@struct.dataclass
class DifferentiableCartesianGenome(CartesianGenome):
    """Cartesian genome augmented with continuous weights and biases.

    Node equation:
        N_i = Activation( sum_k( weights[i,k] * N_{args[i,k]} ) + biases[i] )

    Fields:
        ops:       (num_nodes,)            int32   - activation / opcode index
        args:      (num_nodes, max_arity)  int32   - connection pointers
        out_nodes: (num_outputs,)          int32   - output node pointers
        weights:   (num_nodes, max_arity)  float32 - continuous edge weights
        biases:    (num_nodes,)            float32 - continuous node biases
    """

    weights: chex.Array
    biases: chex.Array

    @property
    def continuous_params(self) -> Tuple[chex.Array, chex.Array]:
        return (self.weights, self.biases)

    @classmethod
    def random_init(  # type: ignore[override]
        cls: Type["DifferentiableCartesianGenome"],
        key: chex.PRNGKey,
        config: DifferentiableCartesianGenomeConfig,
    ) -> "DifferentiableCartesianGenome":
        k_base, k_w, k_b = jax.random.split(key, 3)

        base = CartesianGenome.random_init(k_base, config)

        # Glorot normal initialization for weights
        fan_in = config.max_arity
        std = jnp.sqrt(2.0 / float(max(1, fan_in))) * config.weight_init_std
        weights = jax.random.normal(k_w, (config.num_nodes, config.max_arity)) * std

        # Biases initialized to zero
        biases = jnp.zeros(config.num_nodes, dtype=jnp.float32)

        return cls(
            ops=base.ops,
            args=base.args,
            out_nodes=base.out_nodes,
            weights=weights,
            biases=biases,
        )

    def forget_weights(
        self, key: chex.PRNGKey, config: DifferentiableCartesianGenomeConfig
    ) -> "DifferentiableCartesianGenome":
        """LSMF Forget step: randomizes continuous parameters while preserving topology."""
        k_w, k_b = jax.random.split(key)
        fan_in = config.max_arity
        std = jnp.sqrt(2.0 / float(max(1, fan_in))) * config.weight_init_std
        new_weights = jax.random.normal(k_w, (config.num_nodes, config.max_arity)) * std
        new_biases = jnp.zeros(config.num_nodes, dtype=jnp.float32)
        return cast(
            "DifferentiableCartesianGenome",
            self.replace(weights=new_weights, biases=new_biases),
        )

    def autocorrect(
        self, config: Optional[CartesianGenomeConfig] = None
    ) -> "DifferentiableCartesianGenome":
        """Clamps topological pointers while preserving continuous parameters."""
        if config is None:
            return self

        base_corrected = super().autocorrect(config)
        return cast(
            "DifferentiableCartesianGenome",
            self.replace(
                ops=base_corrected.ops,
                args=base_corrected.args,
                out_nodes=base_corrected.out_nodes,
            ),
        )

    @classmethod
    def create_population(  # type: ignore[override]
        cls: Type["DifferentiableCartesianGenome"],
        key: chex.PRNGKey,
        config: DifferentiableCartesianGenomeConfig,
        size: int,
    ) -> "DifferentiableCartesianGenome":
        keys = jax.random.split(key, size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)


@struct.dataclass
class DifferentiableCartesianPopulation(CartesianPopulation):
    """Population container for batched DifferentiableCartesianGenomes."""

    genes: DifferentiableCartesianGenome
    GENOME_CLS: ClassVar[Type[DifferentiableCartesianGenome]] = DifferentiableCartesianGenome

    @classmethod
    def init(  # type: ignore[override]
        cls: Type["DifferentiableCartesianPopulation"],
        key: chex.PRNGKey,
        config: DifferentiableCartesianGenomeConfig,
        size: int,
    ) -> "DifferentiableCartesianPopulation":
        batched_genes = DifferentiableCartesianGenome.create_population(key, config, size)
        return cls(
            genes=batched_genes,
            fitness=jnp.zeros(size, dtype=jnp.float32),
            config=config,
            info={},
        )
