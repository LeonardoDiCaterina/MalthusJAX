import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Any
from flax import struct
import chex

from malthusjax.core.base import BasePopulation
from malthusjax.engine.emitters.base import BaseEmitter, EmitterState
from malthusjax.core.genome.tensorneat_genome import TensorNeatPopulation, TensorNeatGenome

try:
    from tensorneat.common import State
    from tensorneat.genome import BaseGenome as TNBaseGenome
except ImportError:
    State, TNBaseGenome = Any, Any

@struct.dataclass
class TensorNeatEmitterState(EmitterState):
    tn_state: Any  # TensorNEAT State
    max_node_key: chex.Array
    max_conn_marker: chex.Array

@struct.dataclass
class TensorNeatEmitter(BaseEmitter):
    """
    Native MalthusJAX Quality-Diversity Emitter for TensorNEAT graphs.
    """
    _batch_size: int = struct.field(pytree_node=False)
    genome: Any = struct.field(pytree_node=False)  # The TensorNEAT genome class (e.g. DefaultGenome)
    mutation: Any = struct.field(pytree_node=False, default=None)
    crossover: Any = struct.field(pytree_node=False, default=None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> TensorNeatEmitterState:
        # Initialize a basic TensorNEAT state
        tn_state = State(randkey=key, generation=jnp.float32(0))
        
        # Determine the starting max keys from the initial population
        nodes, conns = initial_population.genes.values
        
        all_nodes_keys = nodes[:, :, 0]
        max_node_key = jnp.max(all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0)
        
        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            all_conns_markers = jax.vmap(self.genome.conn_gene.get_historical_marker, in_axes=(None, 0))(tn_state, conns)
            max_conn_marker = jnp.max(all_conns_markers, where=~jnp.isnan(all_conns_markers), initial=0)
        else:
            max_conn_marker = jnp.array(0.0)
            
        return TensorNeatEmitterState(
            tn_state=tn_state,
            max_node_key=max_node_key,
            max_conn_marker=max_conn_marker
        )

    def ask(self, state: Optional[TensorNeatEmitterState], repertoire: Any, key: chex.Array) -> Tuple[TensorNeatPopulation, TensorNeatEmitterState]:
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # 1. Sample parents from repertoire
        p1_genotypes = repertoire.select(k1, self.batch_size).genotypes
        p2_genotypes = repertoire.select(k2, self.batch_size).genotypes
        
        p1_nodes, p1_conns = p1_genotypes
        p2_nodes, p2_conns = p2_genotypes
        
        cx_keys = jax.random.split(k3, self.batch_size)
        mut_keys = jax.random.split(k4, self.batch_size)
        
        # 2. Crossover
        def _cx(k, n1, c1, n2, c2):
            return self.genome.execute_crossover(state.tn_state, k, n1, c1, n2, c2)
            
        new_nodes, new_conns = jax.vmap(_cx)(cx_keys, p1_nodes, p1_conns, p2_nodes, p2_conns)
        
        # 3. Mutation key preparation
        next_node_key = state.max_node_key + 1
        new_node_keys = jnp.arange(self.batch_size) + next_node_key
        
        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            next_conn_marker = state.max_conn_marker + 1
            new_conn_markers = jnp.arange(self.batch_size * 3).reshape(self.batch_size, 3) + next_conn_marker
        else:
            new_conn_markers = jnp.full((self.batch_size, 3), 0)
            
        # 4. Mutation
        def _mut(k, n, c, nnk, nck):
            return self.genome.execute_mutation(state.tn_state, k, n, c, nnk, nck)
            
        final_nodes, final_conns = jax.vmap(_mut)(mut_keys, new_nodes, new_conns, new_node_keys, new_conn_markers)
        
        # Calculate max keys for next state
        updated_max_node_key = state.max_node_key + self.batch_size
        updated_max_conn_marker = state.max_conn_marker + (self.batch_size * 3)
        
        updated_state = state.replace(
            max_node_key=updated_max_node_key,
            max_conn_marker=updated_max_conn_marker
        )
        
        # 5. Wrap in MalthusJAX Population
        genes = TensorNeatGenome(values=(final_nodes, final_conns))
        offspring_pop = TensorNeatPopulation(
            genes=genes,
            fitness=jnp.full(self.batch_size, -jnp.inf),
            config=None
        )
        
        return offspring_pop, updated_state
