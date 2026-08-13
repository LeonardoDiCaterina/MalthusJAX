from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.operators.base import BaseCrossover, BaseMutation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState


@struct.dataclass
class GeneticMutationEmitter(BaseEmitter):
    """
    Quality-Diversity Emitter that natively wraps a MalthusJAX BaseMutation operator.
    It passes the Engine's generation down to the operator to support dynamic mutation rate schedules.
    """

    mutation: BaseMutation[Any, Any] = struct.field(pytree_node=False)
    genome_config: Any = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.mutation.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "GeneticMutationEmitter":
        return self.replace(_batch_size=length)  # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        return None

    def num_keys_for_sampling(self) -> int:
        return 1

    def num_keys(self) -> int:
        # 1 key for selecting parent + atomic operation keys
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        k1 = keys[0]
        k2 = keys[1:]

        # 1. Sample parents from repertoire
        if hasattr(repertoire, "sample") and not hasattr(repertoire, "select"):
            parent_genotypes, _ = repertoire.sample(k1, self.batch_size)
            if isinstance(parent_genotypes, tuple):
                parent_genotypes = parent_genotypes[0]
        else:
            selection = repertoire.select(k1, self.batch_size)
            parent_genotypes = (
                selection.genes.values if hasattr(selection, "genes") else selection.genotypes
            )

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
        offspring_pop = self.mutation(
            vmap_keys, parent_pop, config=self.genome_config, generation=generation
        )

        return offspring_pop, state


@struct.dataclass
class GeneticCrossoverEmitter(BaseEmitter):
    """
    Quality-Diversity Emitter that natively wraps a MalthusJAX BaseCrossover operator.
    """

    crossover: BaseCrossover[Any, Any] = struct.field(pytree_node=False)
    genome_config: Any = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.crossover.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "GeneticCrossoverEmitter":
        return self.replace(_batch_size=length)  # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        return None

    def num_keys_for_sampling(self) -> int:
        return 2

    def num_keys(self) -> int:
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        """
        Samples parent pairs from the repertoire and executes MalthusJAX Tier-3 crossover.
        """
        k1, k2 = keys[:2], keys[2:]
        k1_a, k1_b = jax.random.split(k1[0])

        # 1. Sample two sets of parents from repertoire
        if hasattr(repertoire, "sample") and not hasattr(repertoire, "select"):
            p1_genotypes, _ = repertoire.sample(k1_a, self.batch_size)
            p2_genotypes, _ = repertoire.sample(k1_b, self.batch_size)
            if isinstance(p1_genotypes, tuple):
                p1_genotypes = p1_genotypes[0]
            if isinstance(p2_genotypes, tuple):
                p2_genotypes = p2_genotypes[0]
        else:
            p1_selection = repertoire.select(k1_a, self.batch_size)
            p2_selection = repertoire.select(k1_b, self.batch_size)
            p1_genotypes = (
                p1_selection.genes.values
                if hasattr(p1_selection, "genes")
                else p1_selection.genotypes
            )
            p2_genotypes = (
                p2_selection.genes.values
                if hasattr(p2_selection, "genes")
                else p2_selection.genotypes
            )

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
        offspring_pop = self.crossover(
            vmap_keys, p1_pop, p2_pop, config=self.genome_config, generation=generation
        )

        return offspring_pop, state


@struct.dataclass
class GeneticMixingEmitter(BaseEmitter):
    """
    Quality-Diversity Emitter that natively combines a BaseCrossover and BaseMutation.
    It executes both operations and uses a mask at the Tier-3 (batch) level to allow XLA
    to fuse the kernel, maximizing SIMT efficiency without relying on AtomicEmitter.
    """

    mutation: BaseMutation[Any, Any] = struct.field(pytree_node=False)
    crossover: BaseCrossover[Any, Any] = struct.field(pytree_node=False)
    variation_percentage: float = struct.field(pytree_node=False)
    genome_config: Any = struct.field(pytree_node=False)
    _batch_size: int = struct.field(pytree_node=False)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return max(
            self.mutation.num_keys_per_atomic_operation,
            self.crossover.num_keys_per_atomic_operation,
        )

    def set_input_length(self, length: int) -> "GeneticMixingEmitter":
        return self.replace(_batch_size=length)  # type: ignore[attr-defined]

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        return None

    def num_keys_for_sampling(self) -> int:
        return 2

    def num_keys(self) -> int:
        # 2 keys for sampling + max required for atomic operations
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        k1_a = keys[0]
        k1_b = keys[1]
        k2 = keys[2:]

        # 1. Sample parents
        if hasattr(repertoire, "sample") and not hasattr(repertoire, "select"):
            p1_genotypes, _ = repertoire.sample(k1_a, self.batch_size)
            p2_genotypes, _ = repertoire.sample(k1_b, self.batch_size)
            if isinstance(p1_genotypes, tuple):
                p1_genotypes = p1_genotypes[0]
            if isinstance(p2_genotypes, tuple):
                p2_genotypes = p2_genotypes[0]
        else:
            p1_selection = repertoire.select(k1_a, self.batch_size)
            p2_selection = repertoire.select(k1_b, self.batch_size)
            p1_genotypes = (
                p1_selection.genes.values
                if hasattr(p1_selection, "genes")
                else p1_selection.genotypes
            )
            p2_genotypes = (
                p2_selection.genes.values
                if hasattr(p2_selection, "genes")
                else p2_selection.genotypes
            )

        p1_pop = self.genome_config.init_population(k1_a, self.batch_size)
        p1_pop = p1_pop.replace(genes=p1_pop.genes.replace(values=p1_genotypes))

        p2_pop = self.genome_config.init_population(k1_b, self.batch_size)
        p2_pop = p2_pop.replace(genes=p2_pop.genes.replace(values=p2_genotypes))

        # 2. Key budgeting
        atomic_keys = self.num_keys_per_atomic_operation
        if getattr(k2, "ndim", 1) > 1:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys, k2.shape[-1])
        else:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys)

        # 3. Apply Tier-3 operators to entire batch
        mut_keys = vmap_keys[:, : self.mutation.num_keys_per_atomic_operation]
        cross_keys = vmap_keys[:, : self.crossover.num_keys_per_atomic_operation]

        mut_pop = self.mutation(mut_keys, p1_pop, config=self.genome_config, generation=generation)
        cross_pop = self.crossover(
            cross_keys, p1_pop, p2_pop, config=self.genome_config, generation=generation
        )

        # 4. Mask and Blend at the array level for XLA fusion
        # QDAX randomly applies crossover with variation_percentage to each individual
        # Wait, QDAX MixingEmitter actually concatenates n_variation and n_mutation!
        # So it is a deterministic split! Reverting to deterministic split to match QDAX.
        n_variation = int(self._batch_size * self.variation_percentage)
        mask = (jnp.arange(self.batch_size) < n_variation).astype(jnp.float32)

        # Expand mask to match gene dims (N, 1, 1...)
        mask_expanded = mask
        for _ in range(mut_pop.genes.values.ndim - 1):
            mask_expanded = jnp.expand_dims(mask_expanded, axis=-1)

        blended_genes = (
            mask_expanded * cross_pop.genes.values + (1.0 - mask_expanded) * mut_pop.genes.values
        )

        final_pop = mut_pop.replace(genes=mut_pop.genes.replace(values=blended_genes))  # type: ignore[attr-defined]
        return final_pop, state
