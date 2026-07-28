from typing import Any, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import struct
import chex

from malthusjax.core.base import BasePopulation
from malthusjax.operators.base import BaseMutation, BaseCrossover
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

@struct.dataclass
class GeneticMutationEmitter(BaseEmitter):
    """
    Quality-Diversity Emitter that natively wraps a MalthusJAX BaseMutation operator.
    It passes the Engine's generation down to the operator to support dynamic mutation rate schedules.
    """
    mutation: BaseMutation = struct.field(pytree_node=False)
    genome_config: Any = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.mutation.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> 'GeneticMutationEmitter':
        return self.replace(_batch_size=length)

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> Optional[EmitterState]:
        return None

    def num_keys_for_sampling(self) -> int:
        return 1

    def num_keys(self) -> int:
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array, generation: int = 0, params: Any = None) -> Tuple[BasePopulation, Optional[EmitterState]]:
        k1 = keys[0]
        k2 = keys[1:]
        
        # 1. Sample parents from repertoire
        parent_genotypes = repertoire.select(k1, self.batch_size).genotypes
        parent_pop = self.genome_config.init_population(k1, self.batch_size)
        parent_pop = parent_pop.replace(genes=parent_pop.genes.replace(values=parent_genotypes))
        
        # 2. Budget keys for mutation
        atomic_keys = self.num_keys_per_atomic_operation
        if getattr(k2, "ndim", 1) > 1:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys, k2.shape[-1])
        else:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys)
            
        # 3. Call the MalthusJAX highly optimized Tier-3 operator
        # We pass generation and params to ensure Gaussian decay schedules work!
        offspring_pop = self.mutation(vmap_keys, parent_pop, config=self.genome_config, generation=generation)
        
        return offspring_pop, state

@struct.dataclass
class GeneticCrossoverEmitter(BaseEmitter):
    """
    Quality-Diversity Emitter that natively wraps a MalthusJAX BaseCrossover operator.
    """
    crossover: BaseCrossover = struct.field(pytree_node=False)
    genome_config: Any = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.crossover.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> 'GeneticCrossoverEmitter':
        return self.replace(_batch_size=length)

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> Optional[EmitterState]:
        return None

    def num_keys_for_sampling(self) -> int:
        return 2

    def num_keys(self) -> int:
        # 2 keys for selecting two sets of parents + atomic operation keys
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array, generation: int = 0, params: Any = None) -> Tuple[BasePopulation, Optional[EmitterState]]:
        k1_a = keys[0]
        k1_b = keys[1]
        k2 = keys[2:]
        
        # 1. Sample two sets of parents from repertoire
        p1_genotypes = repertoire.select(k1_a, self.batch_size).genotypes
        p2_genotypes = repertoire.select(k1_b, self.batch_size).genotypes
        
        p1_pop = self.genome_config.init_population(k1_a, self.batch_size)
        p1_pop = p1_pop.replace(genes=p1_pop.genes.replace(values=p1_genotypes))
        
        p2_pop = self.genome_config.init_population(k1_b, self.batch_size)
        p2_pop = p2_pop.replace(genes=p2_pop.genes.replace(values=p2_genotypes))
        
        # 2. Budget keys for crossover
        atomic_keys = self.num_keys_per_atomic_operation
        if getattr(k2, "ndim", 1) > 1:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys, k2.shape[-1])
        else:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys)
            
        # 3. Call the MalthusJAX highly optimized Tier-3 operator
        offspring_pop = self.crossover(vmap_keys, p1_pop, p2_pop, config=self.genome_config, generation=generation)
        
        return offspring_pop, state
