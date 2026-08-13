from typing import Any, Optional, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
from malthusjax.operators.emitters.base import AtomicEmitter, EmitterState
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitterState

try:
    from tensorneat.common import State
except ImportError:
    State = Any


@struct.dataclass
class TensorNeatMutationEmitter(AtomicEmitter):
    """
    Quality-Diversity Emitter for TensorNEAT graphs that only applies mutation.
    """

    _batch_size: int = struct.field(pytree_node=False)
    genome: Any = struct.field(pytree_node=False)
    mutation: Any = struct.field(pytree_node=False, default=None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def set_input_length(self, length: int) -> "TensorNeatMutationEmitter":
        return self.replace(_batch_size=length)  # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> TensorNeatEmitterState:
        tn_state = State(randkey=key, generation=jnp.float32(0))
        nodes, conns = initial_population.genes.values
        all_nodes_keys = nodes[:, :, 0]
        max_node_key = jnp.max(all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0)

        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            all_conns_markers = jax.vmap(
                self.genome.conn_gene.get_historical_marker, in_axes=(None, 0)
            )(tn_state, conns)
            max_conn_marker = jnp.max(
                all_conns_markers, where=~jnp.isnan(all_conns_markers), initial=0
            )
        else:
            max_conn_marker = jnp.array(0.0)

        return TensorNeatEmitterState(
            tn_state=tn_state, max_node_key=max_node_key, max_conn_marker=max_conn_marker
        )

    def _sample_parents(  # type: ignore[override]
        self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array
    ) -> Tuple[Any, Any]:
        assert state is not None
        tn_state = cast(TensorNeatEmitterState, state)
        p_genotypes = repertoire.select(keys, self.batch_size).genotypes
        p_nodes, p_conns = p_genotypes
        parents_tuple = (p_nodes, p_conns)

        # Batch-level prep for mutation keys
        next_node_key = tn_state.max_node_key + 1
        new_node_keys = jnp.arange(self.batch_size) + next_node_key

        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            next_conn_marker = tn_state.max_conn_marker + 1
            new_conn_markers = (
                jnp.arange(self.batch_size * 3).reshape(self.batch_size, 3) + next_conn_marker
            )
        else:
            new_conn_markers = jnp.full((self.batch_size, 3), 0)

        metadata_dict = {"new_node_key": new_node_keys, "new_conn_markers": new_conn_markers}

        updated_state = tn_state.replace(  # type: ignore[attr-defined]
            max_node_key=tn_state.max_node_key + self.batch_size,
            max_conn_marker=tn_state.max_conn_marker + (self.batch_size * 3),
        )

        return parents_tuple, metadata_dict, updated_state  # type: ignore[return-value]

    def _emit_one(  # type: ignore[override]
        self,
        state: Optional[EmitterState],
        keys: chex.Array,
        p_n: chex.Array,
        p_c: chex.Array,
        new_node_key: chex.Array,
        new_conn_markers: chex.Array,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        # Mutation only
        assert state is not None
        tn_state = cast(TensorNeatEmitterState, state)
        mut_n, mut_c = self.genome.execute_mutation(
            tn_state.tn_state, keys[0], p_n, p_c, new_node_key, new_conn_markers
        )
        return mut_n, mut_c

    def _wrap_population(self, offspring_genes: Tuple[chex.Array, chex.Array]) -> BasePopulation[Any]:
        genes = TensorNeatGenome(values=offspring_genes)
        return TensorNeatPopulation(
            genes=genes, fitness=jnp.full(self.batch_size, -jnp.inf), config=None
        )


@struct.dataclass
class TensorNeatCrossoverEmitter(AtomicEmitter):
    """
    Quality-Diversity Emitter for TensorNEAT graphs that only applies crossover.
    """

    _batch_size: int = struct.field(pytree_node=False)
    genome: Any = struct.field(pytree_node=False)
    crossover: Any = struct.field(pytree_node=False, default=None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def num_keys_for_sampling(self) -> int:
        return 2

    def set_input_length(self, length: int) -> "TensorNeatCrossoverEmitter":
        return self.replace(_batch_size=length)  # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> TensorNeatEmitterState:
        # Same state tracking, though crossover doesn't increment node keys.
        tn_state = State(randkey=key, generation=jnp.float32(0))
        nodes, conns = initial_population.genes.values
        all_nodes_keys = nodes[:, :, 0]
        max_node_key = jnp.max(all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0)

        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            all_conns_markers = jax.vmap(
                self.genome.conn_gene.get_historical_marker, in_axes=(None, 0)
            )(tn_state, conns)
            max_conn_marker = jnp.max(
                all_conns_markers, where=~jnp.isnan(all_conns_markers), initial=0
            )
        else:
            max_conn_marker = jnp.array(0.0)

        return TensorNeatEmitterState(
            tn_state=tn_state, max_node_key=max_node_key, max_conn_marker=max_conn_marker
        )

    def _sample_parents(  # type: ignore[override]
        self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array
    ) -> Tuple[Any, Any]:
        k1, k2 = keys[0], keys[1]

        p1_genotypes = repertoire.select(k1, self.batch_size).genotypes
        p2_genotypes = repertoire.select(k2, self.batch_size).genotypes

        p1_nodes, p1_conns = p1_genotypes
        p2_nodes, p2_conns = p2_genotypes

        parents_tuple = (p1_nodes, p1_conns, p2_nodes, p2_conns)
        metadata_dict: dict[str, Any] = {}

        # Crossover doesn't increment global state keys
        return parents_tuple, metadata_dict, state  # type: ignore[return-value]

    def _emit_one(  # type: ignore[override]
        self,
        state: Optional[EmitterState],
        keys: chex.Array,
        p1_n: chex.Array,
        p1_c: chex.Array,
        p2_n: chex.Array,
        p2_c: chex.Array,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        # Crossover only
        assert state is not None
        tn_state = cast(TensorNeatEmitterState, state)
        cx_n, cx_c = self.genome.execute_crossover(tn_state.tn_state, keys[0], p1_n, p1_c, p2_n, p2_c)
        return cx_n, cx_c

    def _wrap_population(self, offspring_genes: Tuple[chex.Array, chex.Array]) -> BasePopulation[Any]:
        genes = TensorNeatGenome(values=offspring_genes)
        return TensorNeatPopulation(
            genes=genes, fitness=jnp.full(self.batch_size, -jnp.inf), config=None
        )
        genes = TensorNeatGenome(values=offspring_genes)
        return TensorNeatPopulation(
            genes=genes, fitness=jnp.full(self.batch_size, -jnp.inf), config=None
        )
