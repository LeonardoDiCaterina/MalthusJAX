"""Quality-Diversity Emitter for LSP Architectures."""

from typing import Any, Optional, Tuple, Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

# We assume that the mutation and crossover operators used here 
# implement a __call__(keys, population, config) interface, 
# which returns a new mutated/crossed-over population.

@struct.dataclass
class LSPQDEmitter(BaseEmitter):
    """QD Emitter for LSP Architectures.
    
    Samples parents from the MAP-Elites repertoire and applies a hybrid strategy:
    - variation_percentage of the batch undergoes crossover (if variation_op is provided).
    - The remainder (or 100% if no variation_op) undergoes mutation.
    """

    mutation_op: Any = struct.field(pytree_node=False)
    variation_op: Any = struct.field(pytree_node=False, default=None)
    variation_percentage: float = struct.field(pytree_node=False, default=0.5)
    _batch_size: int = struct.field(pytree_node=False, default=100)
    genome_config: Any = struct.field(pytree_node=False, default=None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def num_keys(self) -> int:
        return 1

    def set_input_length(self, length: int) -> "LSPQDEmitter":
        return self.replace(_batch_size=length) # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        return None

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        
        key = keys[0]
        
        if self.variation_op is not None:
            n_variation = int(self._batch_size * self.variation_percentage)
        else:
            n_variation = 0
            
        n_mutation = self._batch_size - n_variation

        x_variation = None
        x_mutation = None

        def _wrap_population(genes: Any) -> BasePopulation[Any]:
            # Dummy wrapper to interface with our operators
            size = jax.tree_util.tree_leaves(genes)[0].shape[0]
            return BasePopulation(genes=genes, fitness=jnp.zeros(size), config=self.genome_config)

        if n_variation > 0:
            sample_key_1, sample_key_2, variation_key = jax.random.split(key, 3)
            # Repertoire.select returns a Repertoire object; we extract .genotypes
            p1_genes = repertoire.select(sample_key_1, n_variation).genotypes
            p2_genes = repertoire.select(sample_key_2, n_variation).genotypes
            
            p1_pop = _wrap_population(p1_genes)
            p2_pop = _wrap_population(p2_genes)
            
            v_keys_total = self.variation_op.num_keys((n_variation,))
            v_keys = jax.random.split(variation_key, v_keys_total)
            x_variation_pop = self.variation_op(v_keys, (p1_pop, p2_pop), self.genome_config, generation)
            x_variation = x_variation_pop.genes

        if n_mutation > 0:
            sample_key, mutation_key = jax.random.split(key)
            p_genes = repertoire.select(sample_key, n_mutation).genotypes
            p_pop = _wrap_population(p_genes)
            
            m_keys_total = self.mutation_op.num_keys((n_mutation,)) if hasattr(self.mutation_op, "num_keys") else self.mutation_op.num_keys_per_atomic_operation * n_mutation
            m_keys = jax.random.split(mutation_key, m_keys_total)
            x_mutation_pop = self.mutation_op(m_keys, p_pop, self.genome_config, generation)
            x_mutation = x_mutation_pop.genes

        # Concatenate
        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree.map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        offspring_pop = _wrap_population(genotypes)
        return offspring_pop, state
