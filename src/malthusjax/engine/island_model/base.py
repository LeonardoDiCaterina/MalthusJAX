from abc import abstractmethod
from typing import Any, Generic, Tuple, TypeVar

import chex
import jax
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.engine.base import AbstractEngine

E = TypeVar("E", bound=AbstractEngine[Any, Any])


@struct.dataclass
class BaseIslandModel(Generic[E]):
    """A Meta-Engine that distributes a standard MalthusJAX Engine across isolated islands.

    This wrapper seamlessly upgrades any 1D `BaseEngine` into a 2D Island Model by
    wrapping the engine's init and step methods in `jax.vmap` across the `num_islands` axis,
    and then applying a customizable topological migration policy.
    """

    engine: E
    num_islands: int = struct.field(pytree_node=False)
    migration_interval: int = struct.field(pytree_node=False)
    num_migrants: int = struct.field(pytree_node=False)

    def init_state(self, key: chex.PRNGKey) -> Any:
        """Initializes the engine perfectly mapped across independent islands."""
        keys = jax.random.split(key, self.num_islands)
        # multi_state contains the 2D population
        multi_state = jax.vmap(self.engine.init_state, in_axes=(0,))(keys)
        return multi_state

    def _island_loop(self, island_state: Any) -> Tuple[Any, Any]:
        """Runs the underlying engine for `migration_interval` generations."""

        def _step(state, _):
            next_state, history = self.engine.step(state)
            return next_state, history

        final_state, history = jax.lax.scan(
            _step, island_state, None, length=self.migration_interval
        )
        return final_state, history

    @abstractmethod
    def migrate(self, key: chex.PRNGKey, multi_pop: BasePopulation[Any]) -> BasePopulation[Any]:
        """Applies the specific topological matrix permutation to swap genomes between islands.

        Must be implemented by concrete subclasses like `RingTopologyIsland`.
        """
        raise NotImplementedError

    def step(self, multi_state: Any) -> Tuple[Any, Any]:
        """Outer generation step (1 step = `migration_interval` local generations)."""
        # 1. Evolve all islands independently in parallel
        evolved_state, history = jax.vmap(self._island_loop, in_axes=(0,))(multi_state)

        # 2. Extract population, migrate, and inject back into state
        # We need a global key for migration. We can take it from the first island's rng_key
        # and split it to ensure diversity.
        migration_key, new_island_0_key = jax.random.split(evolved_state.rng_key[0])
        migrated_pop = self.migrate(migration_key, evolved_state.population)

        # 3. Update the state with the migrated population
        next_state = evolved_state.replace(population=migrated_pop)

        # Replace the first island's key to advance the rng stream
        new_rng_keys = next_state.rng_key.at[0].set(new_island_0_key)
        next_state = next_state.replace(rng_key=new_rng_keys)

        return next_state, history
