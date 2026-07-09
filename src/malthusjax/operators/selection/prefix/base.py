"""Base class for Prefix-Aware Selection Operators.

Unlike standard selection operators that receive a 1D `(pop_size,)` fitness array
and return 1D parent indices, prefix selection operators receive a 2D `(pop_size, L)`
fitness matrix (via the ``PrefixPopulation``) and return a 2D array of shape
``(num_selections, 2)`` representing the ``(parent_idx, prefix_idx)`` pairs.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Any, Generic, Optional, Tuple, TypeVar

import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.prefix.population import PrefixPopulation

C = TypeVar("C")


@struct.dataclass
class BasePrefixSelection(Generic[C]):
    """Stateless selection operator for prefix-aware fitness sampling.

    Extracts the `prefix_fitness` matrix from a `PrefixPopulation` and performs
    selection across the flattened `(pop_size * L)` candidate pool.

    Shape contracts:
    - Input: `PrefixPopulation` containing `prefix_fitness` `(pop_size, L)`.
    - Output parents: `(num_selections, 2)` integer array where `output[:, 0]`
      is the genome index and `output[:, 1]` is the prefix row index.
    - Output elites: `(n_elites, 2)` integer array.
    """

    num_selections: int = struct.field(pytree_node=False)
    input_length: int = struct.field(pytree_node=False, default=-1)
    typed_keys: bool = struct.field(pytree_node=False, default=False)
    n_elites: int = struct.field(pytree_node=False, default=0)

    def set_input_length(self, length: int) -> "BasePrefixSelection[C]":
        """Lock population size for static budgeting."""
        return dataclasses.replace(self, input_length=length)

    def set_typed_keys(self, typed: bool) -> "BasePrefixSelection[C]":
        return dataclasses.replace(self, typed_keys=typed)

    def set_n_elites(self, n: int) -> "BasePrefixSelection[C]":
        return dataclasses.replace(self, n_elites=n)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError  # pragma: no cover

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return self.num_keys_per_atomic_operation

    @abstractmethod
    def _select_prefix(
        self, keys: chex.Array, prefix_fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Select parent and prefix indices from a 2D fitness matrix.

        Returns:
            Integer array of shape `(num_selections, 2)`.
        """
        raise NotImplementedError  # pragma: no cover

    def get_elite_indices(self, prefix_fitness: chex.Array, maximize: bool = False) -> chex.Array:
        """Return the `(parent_idx, prefix_idx)` of the top `n_elites` individuals.

        By default, we flatten the `(pop_size, L)` matrix, find the absolute best,
        and unflatten the indices.
        """
        if self.n_elites == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)
            
        pop_size, L = prefix_fitness.shape
        flat_fitness = prefix_fitness.reshape(-1)
        total_candidates = pop_size * L

        if self.n_elites >= total_candidates:
            # Return all pairs
            p_idx, l_idx = jnp.divmod(jnp.arange(total_candidates, dtype=jnp.int32), L)
            return jnp.stack([p_idx, l_idx], axis=-1)

        if maximize:
            # Maximize: we want the *largest* values, so partition on negative
            flat_elite_idx = jnp.argpartition(-flat_fitness, self.n_elites)[: self.n_elites]
        else:
            # Minimize: we want the *smallest* values
            flat_elite_idx = jnp.argpartition(flat_fitness, self.n_elites)[: self.n_elites]

        # Unflatten indices
        parent_idx, prefix_idx = jnp.divmod(flat_elite_idx, L)
        return jnp.stack([parent_idx, prefix_idx], axis=-1)

    def select_with_provenance(
        self, keys: chex.Array, population: PrefixPopulation, config: Optional[C] = None, **kwargs: Any
    ) -> Tuple[chex.Array, chex.Array]:
        """Run a selection pass returning full `(idx, 2)` provenance pairs for research.

        Returns:
            Tuple of `(parent_pairs, elite_pairs)` where each pair is `(genome_idx, prefix_idx)`.
        """
        if population.prefix_fitness is None:
            raise ValueError(
                "PrefixPopulation is missing prefix_fitness. Ensure it was "
                "evaluated by a BasePrefixEvaluator."
            )

        parent_pairs = self._select_prefix(keys, population.prefix_fitness, config, **kwargs)
        
        # We need to know if we are maximizing for get_elite_indices.
        # Fall back to False (minimize) if config doesn't have it, as that's MalthusJAX's default.
        maximize = getattr(config, "maximize", False) if config is not None else False
        elite_pairs = self.get_elite_indices(population.prefix_fitness, maximize=maximize)
        
        return parent_pairs, elite_pairs

    def __call__(
        self, keys: chex.Array, population: PrefixPopulation, config: Optional[C] = None, **kwargs: Any
    ) -> Tuple[chex.Array, chex.Array]:
        """Engine-compatible selection returning 1D parent and elite indices.
        
        This wraps `select_with_provenance` but extracts only the `genome_idx` column
        to remain 100% compatible with the standard `BaseSelection` contract.
        """
        parent_pairs, elite_pairs = self.select_with_provenance(keys, population, config, **kwargs)
        
        # Extract 1D genome indices
        parent_idx = parent_pairs[:, 0]
        
        if elite_pairs.shape[0] > 0:
            elite_idx = elite_pairs[:, 0]
        else:
            elite_idx = jnp.zeros(0, dtype=jnp.int32)
            
        return parent_idx, elite_idx
