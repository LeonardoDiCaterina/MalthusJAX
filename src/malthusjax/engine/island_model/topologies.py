from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.engine.base import AbstractEngine
from malthusjax.engine.island_model.base import BaseIslandModel

E = TypeVar("E", bound=AbstractEngine[Any, Any])


@struct.dataclass
class RingTopologyIsland(BaseIslandModel[E]):
    """Standard GP distributed topology.

    Extracts the top `num_migrants` elites from each island, and shifts them
    exactly one island to the right, injecting them into the worst spots.
    """

    def migrate(self, key: chex.PRNGKey, multi_pop: BasePopulation[Any]) -> BasePopulation[Any]:
        # multi_pop.fitness shape: (num_islands, island_size)
        sort_fitness = -multi_pop.fitness if self.maximize else multi_pop.fitness
        sorted_indices = jnp.argsort(sort_fitness, axis=-1)

        # elite_indices will correspond to the lowest values in sort_fitness, 
        # which are the best individuals for both minimize and maximize.
        elite_indices = sorted_indices[:, : self.num_migrants]
        worst_indices = sorted_indices[:, -self.num_migrants :]

        def gather_elites(arr, idx):
            return arr[idx]

        elites_genes = jax.tree_util.tree_map(
            lambda arr: jax.vmap(gather_elites)(arr, elite_indices), multi_pop.genes
        )
        elites_fitness = jax.vmap(gather_elites)(multi_pop.fitness, elite_indices)

        # 2. Shift elites one island to the right
        migrant_genes = jax.tree_util.tree_map(
            lambda arr: jnp.roll(arr, shift=1, axis=0), elites_genes
        )
        migrant_fitness = jnp.roll(elites_fitness, shift=1, axis=0)

        # 3. Inject migrants into the worst indices
        def inject_migrants(arr, idx, migrants):
            return arr.at[idx].set(migrants)

        final_genes = jax.tree_util.tree_map(
            lambda arr, m_arr: jax.vmap(inject_migrants)(arr, worst_indices, m_arr),
            multi_pop.genes,
            migrant_genes,
        )
        final_fitness = jax.vmap(inject_migrants)(multi_pop.fitness, worst_indices, migrant_fitness)

        return multi_pop.replace(genes=final_genes, fitness=final_fitness)  # type: ignore[attr-defined]


@struct.dataclass
class FullyConnectedIsland(BaseIslandModel[E]):
    """Highly disruptive topology for maximum diffusion.

    Extracts the top elites from all islands into a shared pool, perfectly
    shuffles them, and redistributes them randomly across all islands.
    """

    def migrate(self, key: chex.PRNGKey, multi_pop: BasePopulation[Any]) -> BasePopulation[Any]:
        sort_fitness = -multi_pop.fitness if self.maximize else multi_pop.fitness
        sorted_indices = jnp.argsort(sort_fitness, axis=-1)

        elite_indices = sorted_indices[:, : self.num_migrants]
        worst_indices = sorted_indices[:, -self.num_migrants :]

        def gather_elites(arr, idx):
            return arr[idx]

        elites_genes = jax.tree_util.tree_map(
            lambda arr: jax.vmap(gather_elites)(arr, elite_indices), multi_pop.genes
        )
        elites_fitness = jax.vmap(gather_elites)(multi_pop.fitness, elite_indices)

        # 2. Shuffle all elites globally
        # Reshape to a flat pool of shape (num_islands * num_migrants, ...)
        flat_genes = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), elites_genes)
        flat_fitness = elites_fitness.reshape(-1)

        shuffled_indices = jax.random.permutation(key, flat_fitness.shape[0])

        shuffled_genes = jax.tree_util.tree_map(lambda x: x[shuffled_indices], flat_genes)
        shuffled_fitness = flat_fitness[shuffled_indices]

        # 3. Reshape back and inject
        def inject_shuffled(arr, idx, shuffled_pool, island_idx):
            # Extract exactly num_migrants from the pool for this island
            start = island_idx * self.num_migrants
            island_migrants = jax.lax.dynamic_slice_in_dim(
                shuffled_pool, start, self.num_migrants, axis=0
            )
            return arr.at[idx].set(island_migrants)

        island_indices = jnp.arange(self.num_islands)
        final_genes = jax.tree_util.tree_map(
            lambda arr, s_arr: jax.vmap(inject_shuffled, in_axes=(0, 0, None, 0))(
                arr, worst_indices, s_arr, island_indices
            ),
            multi_pop.genes,
            shuffled_genes,
        )
        final_fitness = jax.vmap(inject_shuffled, in_axes=(0, 0, None, 0))(
            multi_pop.fitness, worst_indices, shuffled_fitness, island_indices
        )

        return multi_pop.replace(genes=final_genes, fitness=final_fitness)  # type: ignore[attr-defined]
