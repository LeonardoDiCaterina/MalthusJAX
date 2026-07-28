from typing import Any, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import struct
import chex

from malthusjax.core.base import BasePopulation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

@struct.dataclass
class MixingEmitterState(EmitterState):
    """
    Composite state holding states for two sub-emitters.
    """
    state_a: Optional[EmitterState]
    state_b: Optional[EmitterState]

@struct.dataclass
class MixingEmitter(BaseEmitter):
    """
    A Compositional Manager that distributes a batch across two sub-emitters.
    """
    emitter_a: BaseEmitter = struct.field(pytree_node=False)
    emitter_b: BaseEmitter = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self.emitter_a.batch_size + self.emitter_b.batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def num_keys(self) -> int:
        # Sum of sub-emitter key requirements
        return self.emitter_a.num_keys() + self.emitter_b.num_keys()

    def set_input_length(self, length: int) -> 'MixingEmitter':
        # Length implies new total batch size. MixingEmitter doesn't easily reshape itself automatically 
        # unless ratio is tracked. The best practice is to set input lengths of sub-emitters directly during init.
        raise NotImplementedError("Set input length on sub-emitters instead of MixingEmitter.")

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> MixingEmitterState:
        k1, k2 = jax.random.split(key)
        state_a = self.emitter_a.init(k1, initial_population, params)
        state_b = self.emitter_b.init(k2, initial_population, params)
        return MixingEmitterState(state_a=state_a, state_b=state_b)

    def ask(self, state: Optional[MixingEmitterState], repertoire: Any, keys: chex.Array, generation: int = 0, params: Any = None) -> Tuple[BasePopulation, Optional[MixingEmitterState]]:
        keys_a = self.emitter_a.num_keys()
        k1 = keys[:keys_a]
        k2 = keys[keys_a:]
        
        state_a_in = state.state_a if state else None
        state_b_in = state.state_b if state else None
        
        pop_a, next_state_a = self.emitter_a.ask(state_a_in, repertoire, k1, generation, params)
        pop_b, next_state_b = self.emitter_b.ask(state_b_in, repertoire, k2, generation, params)
        
        # Merge populations. This assumes pop_a and pop_b have identical pytree structures (which they must).
        merged_genes = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), pop_a.genes, pop_b.genes)
        
        merged_info = dict(pop_a.info)
        if "descriptors" in pop_a.info and "descriptors" in pop_b.info:
            merged_info["descriptors"] = jnp.concatenate([pop_a.info["descriptors"], pop_b.info["descriptors"]], axis=0)
            
        merged_pop = pop_a.replace(
            genes=merged_genes,
            fitness=jnp.concatenate([pop_a.fitness, pop_b.fitness], axis=0),
            info=merged_info
        )
        
        return merged_pop, MixingEmitterState(state_a=next_state_a, state_b=next_state_b)

    def tell(
        self, 
        state: Optional[MixingEmitterState], 
        repertoire: Any,
        population: BasePopulation, 
        fitnesses: chex.Array, 
        descriptors: chex.Array,
        key: chex.Array
    ) -> Optional[MixingEmitterState]:
        k1, k2 = jax.random.split(key)
        
        # Split populations back
        size_a = self.emitter_a.batch_size
        
        pop_a_genes = jax.tree_util.tree_map(lambda x: x[:size_a], population.genes)
        pop_b_genes = jax.tree_util.tree_map(lambda x: x[size_a:], population.genes)
        
        pop_a = population.replace(genes=pop_a_genes, fitness=fitnesses[:size_a])
        pop_b = population.replace(genes=pop_b_genes, fitness=fitnesses[size_a:])
        
        state_a_in = state.state_a if state else None
        state_b_in = state.state_b if state else None
        
        desc_a = descriptors[:size_a] if descriptors is not None else None
        desc_b = descriptors[size_a:] if descriptors is not None else None
        
        next_state_a = self.emitter_a.tell(state_a_in, repertoire, pop_a, fitnesses[:size_a], desc_a, k1)
        next_state_b = self.emitter_b.tell(state_b_in, repertoire, pop_b, fitnesses[size_a:], desc_b, k2)
        
        return MixingEmitterState(state_a=next_state_a, state_b=next_state_b)
