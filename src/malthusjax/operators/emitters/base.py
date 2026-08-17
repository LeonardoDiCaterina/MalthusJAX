from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import chex
import jax
from flax import struct

from malthusjax.core.base import BasePopulation


@struct.dataclass
class EmitterState:
    """Base class for any emitter-specific state (e.g. tracking markers, running means)."""

    pass


class BaseEmitter(ABC):
    """
    Abstract Base Class for Quality-Diversity Emitters in MalthusJAX.

    This acts as the Compositional interface and the ResourceMapper endpoint.
    Compositional emitters (e.g., MixingEmitter) can inherit directly from this
    class and aggregate the `num_keys` of their sub-emitters.
    """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The number of genomes generated per generation cycle."""
        pass

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Keys required per atomic generation (for ResourceMapper static allocation)."""
        pass

    def num_keys(self) -> int:
        """
        Total keys needed for the entire batch.
        Returns total number of keys for ResourceMapper pre-allocation.
        """
        return self.batch_size * self.num_keys_per_atomic_operation

    @abstractmethod
    def set_input_length(self, length: int) -> "BaseEmitter":
        """Lock the batch size for static key budgeting."""
        pass

    @abstractmethod
    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        """Initializes any required internal state using the initial population."""
        pass

    @abstractmethod
    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        """
        Samples parents from the repertoire and generates a batch of mutated offspring.
        Receives a pre-allocated flat buffer of keys from the ResourceMapper.
        """
        pass

    def tell(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        population: BasePopulation[Any],
        fitnesses: chex.Array,
        descriptors: chex.Array,
        key: chex.Array,
    ) -> Optional[EmitterState]:
        """
        Updates the internal Emitter state using the evaluated population metrics.
        """
        return state


class AtomicEmitter(BaseEmitter):
    """
    Base class for Emitters that actually execute genetic or gradient-based variation.
    It forces the strict 3-Tier 'Single Consumer' architecture.
    """

    @abstractmethod
    def _sample_parents(
        self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array
    ) -> Tuple[Any, Any]:
        """
        Tier 2 - Sampling & Data Prep.
        Returns (batched_parents, updated_emitter_state_or_metadata).
        """
        pass

    @abstractmethod
    def _emit_one(
        self, state: Optional[EmitterState], key: chex.Array, *parents: Any, **kwargs: Any
    ) -> Any:
        """
        Tier 1 - Pure atomic generation for a single offspring instance.
        """
        pass

    def num_keys_for_sampling(self) -> int:
        """Keys needed for Tier 2 sampling."""
        return 1

    def num_keys(self) -> int:
        """
        Total keys: keys for sampling parents + keys for atomic emission.
        """
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
        Tier 3 - Orchestrator.
        Receives pre-allocated keys block, samples parents, and vmaps the atomic generation.
        """
        num_sample = self.num_keys_for_sampling()
        k_sample = keys[:num_sample] if num_sample > 1 else keys[0]
        k2 = keys[num_sample:]

        # 1. Tier 2: Sample parents
        # (parents can be a tuple of PyTrees for crossover, or a single PyTree for mutation)
        # metadata can contain batched variables (e.g. node_keys for TensorNEAT)
        parents, metadata, new_state = self._sample_parents(state, repertoire, k_sample)  # type: ignore[misc]

        atomic_keys = self.num_keys_per_atomic_operation

        # Handle JAX legacy keys (N, 2) vs typed keys (N,)
        if getattr(k2, "ndim", 1) > 1:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys, k2.shape[-1])
        else:
            vmap_keys = k2.reshape(self.batch_size, atomic_keys)

        # 3. Tier 1: Vmap atomic emission
        def _vmap_emit(k, p, meta):
            return self._emit_one(new_state, k, p, **meta)

        # Assumes parents is a tuple/list to unpack. If not, it just passes it.
        # We need a robust way to vmap over varying numbers of parent arguments.
        # For simplicity, we assume `parents` is a dictionary or a single pytree that the subclass unpacks inside `_emit_one`!
        # Actually, let's assume `parents` is a tuple of PyTrees representing arguments.
        if not isinstance(parents, tuple):
            parents = (parents,)

        def _vectorized_emit(k_block, *parent_args, **meta_args):
            return self._emit_one(new_state, k_block, *parent_args, **meta_args)

        # Using jax.vmap. Note: meta is expected to be a dict of batched variables.
        # We can just vmap over the dictionary as well.
        # However, jax.vmap takes in_axes.
        # A simple pattern: pass a single 'parents_tuple' and 'metadata_dict' to _emit_one.

        def _wrapper(k, p_tuple, m_dict):
            return self._emit_one(new_state, k, *p_tuple, **m_dict)

        offspring_genes = jax.vmap(_wrapper)(vmap_keys, parents, metadata)

        # 4. Wrap in Population
        # Note: The subclass is responsible for defining the specific Population type.
        # We can require a `_wrap_population` method.
        offspring_pop = self._wrap_population(offspring_genes)

        return offspring_pop, new_state

    @abstractmethod
    def _wrap_population(self, offspring_genes: Any) -> BasePopulation[Any]:
        """Wraps the vmap'd genes into a BasePopulation[Any]."""
        pass
