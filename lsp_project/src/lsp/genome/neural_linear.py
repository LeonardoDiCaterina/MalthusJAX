"""Differentiable Multi Expression Programming (dMEP) Genome.

Combines the prefix-aware linear sequence evaluation of MEP with the
continuous weights and biases of dCGPANN.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.linear import BasePrefixAwareGenome, PrefixGenomeConfig

from malthusjax.core.base import BasePopulation


@struct.dataclass
class NeuralPrefixGenomeConfig(PrefixGenomeConfig):
    """Configuration for Neural Prefix Aware Genomes (dMEP)."""

    num_outputs: int = struct.field(pytree_node=False, default=1)
    mep_output_strategy: str = struct.field(pytree_node=False, default="dynamic")

    @property
    def dtype(self) -> Any:
        return jnp.float32

    def init_population(self, key: chex.PRNGKey, size: int) -> "NeuralPrefixPopulation":
        from lsp.genome.neural_linear import NeuralPrefixPopulation
        return NeuralPrefixPopulation.init_random(key, self, size)


@struct.dataclass
class NeuralPrefixGenome(BasePrefixAwareGenome):
    """Neural Linear Genome extended with continuous parameters."""

    weights: chex.Array
    biases: chex.Array
    out_nodes: chex.Array | None = None
    readout_weights: chex.Array | None = None
    readout_biases: chex.Array | None = None

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: NeuralPrefixGenomeConfig) -> "NeuralPrefixGenome":
        key, k_ops, k_args, k_w, k_b, k_out, k_rw, k_rb = jax.random.split(key, 8)

        L = config.length
        N = config.num_inputs
        max_arity = config.max_arity

        ops = jax.random.randint(k_ops, shape=(L,), minval=0, maxval=config.num_ops)

        u = jax.random.uniform(k_args, shape=(L, max_arity))
        max_vals = jnp.arange(N, N + L, dtype=jnp.float32)[:, None]
        args = jnp.floor(u * max_vals).astype(jnp.int32)

        stddev = jnp.sqrt(2.0 / max_arity)
        weights = jax.random.normal(k_w, shape=(L, max_arity)) * stddev
        biases = jax.random.normal(k_b, shape=(L,)) * stddev

        out_nodes = None
        readout_weights = None
        readout_biases = None

        if config.mep_output_strategy == "out_nodes":
            out_nodes = jax.random.randint(k_out, shape=(config.num_outputs,), minval=0, maxval=L)
        elif config.mep_output_strategy == "linear_readout":
            ro_stddev = jnp.sqrt(2.0 / L)
            readout_weights = jax.random.normal(k_rw, shape=(L, config.num_outputs)) * ro_stddev
            readout_biases = jnp.zeros((config.num_outputs,), dtype=jnp.float32)

        return cls(
            ops=ops,
            args=args,
            weights=weights,
            biases=biases,
            out_nodes=out_nodes,
            readout_weights=readout_weights,
            readout_biases=readout_biases
        )

@struct.dataclass
class NeuralPrefixPopulation(BasePopulation[NeuralPrefixGenome]):
    genes: NeuralPrefixGenome
    fitness: chex.Array
    config: NeuralPrefixGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: NeuralPrefixGenomeConfig, size: int) -> "NeuralPrefixPopulation":
        keys = jax.random.split(key, size)
        batched_genes = jax.vmap(NeuralPrefixGenome.random_init, in_axes=(0, None))(keys, config)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)

__all__ = ["NeuralPrefixGenomeConfig", "NeuralPrefixGenome", "NeuralPrefixPopulation"]
