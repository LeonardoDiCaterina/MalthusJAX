"""Tensorized NEAT Genome for MalthusJAX.

Wraps TensorNEAT's fixed-capacity (nodes, conns) matrices with full BaseGenome
compliance, JIT-compatible masking, structural distance, and population generation.
"""

from __future__ import annotations

from typing import Any, ClassVar, Tuple, Type

import chex
import jax
import jax.numpy as jnp
from flax import struct

from tensorneat.common import State
from tensorneat.genome import DefaultGenome, RecurrentGenome

from malthusjax.composer import register_genome
from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.genomes.neat_genome")

_field: Any = struct.field


@register_genome("neat_genome", override=True)
@register_genome("neat", override=True)
@struct.dataclass
class NeatGenomeConfig:
    """Configuration for tensorized NEAT genomes.

    Attributes:
        num_inputs: Number of network input dimensions.
        num_outputs: Number of network output dimensions.
        max_nodes: Fixed buffer capacity for maximum nodes (XLA static shape).
        max_conns: Fixed buffer capacity for maximum connections (XLA static shape).
        recurrent: Whether recurrent connections are permitted.
        output_transform: Optional activation/transform on output nodes.
    """

    num_inputs: int = _field(pytree_node=False, default=2)
    num_outputs: int = _field(pytree_node=False, default=1)
    max_nodes: int = _field(pytree_node=False, default=20)
    max_conns: int = _field(pytree_node=False, default=40)
    recurrent: bool = _field(pytree_node=False, default=False)
    node_gene: Any = _field(pytree_node=False, default=None)
    conn_gene: Any = _field(pytree_node=False, default=None)
    mutation: Any = _field(pytree_node=False, default=None)
    crossover: Any = _field(pytree_node=False, default=None)
    distance: Any = _field(pytree_node=False, default=None)

    def to_tensorneat_genome(self) -> Any:
        """Instantiates the underlying TensorNEAT genome engine."""
        cls = RecurrentGenome if self.recurrent else DefaultGenome
        kwargs: dict[str, Any] = {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "max_nodes": self.max_nodes,
            "max_conns": self.max_conns,
        }
        if self.node_gene is not None:
            kwargs["node_gene"] = self.node_gene
        if self.conn_gene is not None:
            kwargs["conn_gene"] = self.conn_gene
        if self.mutation is not None:
            kwargs["mutation"] = self.mutation
        if self.crossover is not None:
            kwargs["crossover"] = self.crossover
        if self.distance is not None:
            kwargs["distance"] = self.distance
        return cls(**kwargs)

    @property
    def dtype(self) -> Any:
        """Data type of the genome tensors."""
        return jnp.float32

    def init_population(self, key: chex.PRNGKey, size: int) -> "NeatPopulation":
        """Initializes a population container of batched NeatGenomes."""
        return NeatPopulation.init(key, self, size)


@struct.dataclass
class NeatGenome(BaseGenome):  # type: ignore[misc]
    """JAX-native NEAT Genome containing fixed-capacity node and connection matrices.

    Attributes:
        nodes: Shape (max_nodes, 5) float32 matrix. Inactive rows contain NaN.
        conns: Shape (max_conns, 3) float32 matrix. Inactive rows contain NaN.
    """

    nodes: chex.Array
    conns: chex.Array

    @property
    def values(self) -> Tuple[chex.Array, chex.Array]:
        """Returns the (nodes, conns) tuple representing the chromosome."""
        return (self.nodes, self.conns)

    @property
    def size(self) -> int:
        """Total number of scalar parameters in the genome buffer."""
        return int(self.nodes.size + self.conns.size)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Buffer shape descriptor (max_nodes, max_conns)."""
        return (self.nodes.shape[0], self.conns.shape[0])

    def get_active_mask(self) -> Tuple[chex.Array, chex.Array]:
        """Returns boolean masks (active_nodes, active_conns).

        A node/conn is active if its first attribute is finite (not NaN).
        """
        active_nodes = ~jnp.isnan(self.nodes[:, 0])
        active_conns = ~jnp.isnan(self.conns[:, 0])
        return active_nodes, active_conns

    @property
    def num_active_nodes(self) -> chex.Array:
        """Number of active nodes participating in the network."""
        return jnp.sum(~jnp.isnan(self.nodes[:, 0]))

    @property
    def num_active_conns(self) -> chex.Array:
        """Number of active synapses participating in the network."""
        return jnp.sum(~jnp.isnan(self.conns[:, 0]))

    @property
    def connection_density(self) -> chex.Array:
        """Ratio of active connections to maximum connection capacity in [0, 1]."""
        return self.num_active_conns / self.conns.shape[0]

    @classmethod
    def random_init(
        cls: Type["NeatGenome"], key: chex.PRNGKey, config: NeatGenomeConfig
    ) -> "NeatGenome":
        """Generates a randomly initialized initial NEAT genome with direct input-output wiring."""
        tn_genome = config.to_tensorneat_genome()
        state = State().register(randkey=key)
        state = tn_genome.setup(state)
        nodes, conns = tn_genome.initialize(state, key)
        return cls(nodes=nodes, conns=conns)

    @classmethod
    def create_population(
        cls: Type["NeatGenome"],
        key: chex.PRNGKey,
        config: NeatGenomeConfig,
        size: int,
    ) -> "NeatGenome":
        """Vectorized generation of an initial population of genomes."""
        keys = jax.random.split(key, size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)

    def autocorrect(self, config: Any = None) -> "NeatGenome":
        """Identity autocorrect for compliant BaseGenome contract."""
        return self

    def distance(self, other: BaseGenome, metric: str = "euclidean") -> chex.Array:
        """Calculates structural and synaptic weight distance between two genomes."""
        if not isinstance(other, NeatGenome):
            return jnp.array(0.0)

        # 1. Active connection overlap & weight differences
        c1_active = ~jnp.isnan(self.conns[:, 0])
        c2_active = ~jnp.isnan(other.conns[:, 0])
        both_active = c1_active & c2_active

        # Weight is column index 0 or 2 depending on TensorNEAT conn_gene
        w1 = jnp.where(both_active, self.conns[:, 0], 0.0)
        w2 = jnp.where(both_active, other.conns[:, 0], 0.0)
        diff = jnp.sum(jnp.abs(w1 - w2))

        # Disjoint count
        disjoint = jnp.sum(c1_active ^ c2_active)
        return jnp.array(disjoint + diff, dtype=jnp.float32)

    @classmethod
    def from_tensor(cls: Type["NeatGenome"], arr: Any, config: Any = None) -> "NeatGenome":
        """Constructs a NeatGenome from a (nodes, conns) tuple or array."""
        if isinstance(arr, tuple) and len(arr) == 2:
            return cls(nodes=arr[0], conns=arr[1])
        if isinstance(arr, NeatGenome):
            return arr
        raise ValueError("NeatGenome requires (nodes, conns) tuple")


@struct.dataclass
class NeatPopulation(BasePopulation[NeatGenome]):  # type: ignore[misc]
    """Population container for batched NeatGenomes."""

    genes: NeatGenome
    GENOME_CLS: ClassVar[Type[NeatGenome]] = NeatGenome

    @classmethod
    def init(
        cls: Type["NeatPopulation"],
        key: chex.PRNGKey,
        config: NeatGenomeConfig,
        size: int,
    ) -> "NeatPopulation":
        batched_genes = NeatGenome.create_population(key, config, size)
        return cls(
            genes=batched_genes,
            fitness=jnp.zeros(size, dtype=jnp.float32),
            config=config,
            info={},
        )


__all__ = ["NeatGenome", "NeatGenomeConfig", "NeatPopulation"]
