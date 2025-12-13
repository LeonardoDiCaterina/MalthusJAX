"""
Linear genome crossover operators.

Implements crossover operators tailored for linear genomes
that preserve topological validity.
"""

from typing import TypeVar
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation

# Generic Types
G = TypeVar("G", bound="LinearGenome")  # Genome Type
C = TypeVar("C")  # Config Type


@struct.dataclass
class LinearCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig, LinearPopulation]):
    """
    Linear GP uniform crossover operator.
    
    Performs coin-flip mixing of operation codes and arguments
    between two parent genomes.
    """
    # Dynamic parameters
    mixing_ratio: float = 0.5  # Probability to take from parent1 vs parent2
    
    
    def __call__(self, key: chex.PRNGKey, pop1: G, pop2: G, config: C) -> G:
        """
        Handles the FULL population batching internally.
        
        Input Shapes:
          pop1: (Pop_Size, Genome_Len, 3) -> (177, 73, 3)
          pop2: (Pop_Size, Genome_Len, 3) -> (177, 73, 3)
        
        Output Shape:
          (Pop_Size, Num_Offspring, Genome_Len, 3)
        """
        # 1. Determine Population Size from input (e.g., 177)
        pop_size = pop1.ops.shape[0]
        
        # 2. Split keys for the population (1 key per parent pair)
        keys = jax.random.split(key, pop_size)

        # 3. Define the Single-Pair Logic (Closure)
        #    This function takes ONE pair of parents and returns N children
        def process_single_pair(k, p1, p2):
            # Split key for the multiple children (Inner Batching)
            child_keys = jax.random.split(k, self.num_offspring)
            
            # Inner Vmap: Generate 'num_offspring' children from this pair
            children = jax.vmap(
                lambda ck, a, b: self._cross_one(ck, a, b, config),
                in_axes=(0, None, None) 
            )(child_keys, p1, p2)
            
            return children

        # 4. Outer Vmap: Apply logic across the population
        #    Iterate over keys(0), pop1(0), pop2(0), keep config(None)
        offspring_batch = jax.vmap(process_single_pair, in_axes=(0, 0, 0))(keys, pop1, pop2)

        # 5. Optional: Squeeze if you only want 1 child per parent
        #    Changes (177, 1, 73, 3) -> (177, 73, 3)
        if self.num_offspring == 1:
            return jnp.squeeze(offspring_batch, axis=1)
            
        return offspring_batch
    

    def _cross_one(self, key: chex.PRNGKey, parent1: LinearGenome, parent2: LinearGenome, 
                  config: LinearGenomeConfig) -> LinearGenome:
        """Apply crossover to produce one offspring."""
        # Generate mixing mask: True = take from parent1, False = take from parent2
        mask = jax.random.bernoulli(key, self.mixing_ratio, parent1.ops.shape)

        # Mix operation codes
        child_ops = jnp.where(mask, parent1.ops, parent2.ops)

        # Mix arguments (broadcast mask to match argument dimensions)
        mask_expanded = mask[:, None]  # Shape (L,) -> (L, 1)
        child_args = jnp.where(mask_expanded, parent1.args, parent2.args)

        # Uniform crossover between topologically valid parents produces valid offspring
        # No autocorrect needed
        return parent1.replace(ops=child_ops, args=child_args)