from abc import abstractmethod
from typing import Any, Generic, Tuple, TypeVar

import chex
import jax
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.engine.base import AbstractEngine

E = TypeVar("E", bound=AbstractEngine)

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

    def init(self, key: chex.PRNGKey, config: Any, island_size: int) -> Tuple[BasePopulation, Any]:
        """Initializes the engine perfectly mapped across independent islands."""
        keys = jax.random.split(key, self.num_islands)
        # multi_pop shape: (num_islands, island_size, ...)
        multi_pop, multi_state = jax.vmap(self.engine.init, in_axes=(0, None, None))(
            keys, config, island_size
        )
        return multi_pop, multi_state

    def _island_loop(self, island_pop: BasePopulation, island_state: Any, r_key: chex.PRNGKey, global_gen_start: int) -> Tuple[BasePopulation, Any]:
        """Runs the underlying engine for `migration_interval` generations."""
        def _step(carry, _):
            pop, state, key, gen = carry
            k_step, next_key = jax.random.split(key)
            next_pop, next_state = self.engine.step(k_step, pop, state, gen)
            return (next_pop, next_state, next_key, gen + 1), None
            
        carry = (island_pop, island_state, r_key, global_gen_start)
        carry, _ = jax.lax.scan(_step, carry, None, length=self.migration_interval)
        final_pop, final_state = carry[0], carry[1]
        
        return final_pop, final_state

    @abstractmethod
    def migrate(self, key: chex.PRNGKey, multi_pop: BasePopulation) -> BasePopulation:
        """Applies the specific topological matrix permutation to swap genomes between islands.
        
        Must be implemented by concrete subclasses like `RingTopologyIsland`.
        """
        raise NotImplementedError
        
    def step(self, key: chex.PRNGKey, multi_pop: BasePopulation, multi_state: Any, global_gen_start: int) -> Tuple[BasePopulation, Any]:
        """Outer generation step (1 step = `migration_interval` local generations)."""
        k_loop, k_migrate = jax.random.split(key)
        keys = jax.random.split(k_loop, self.num_islands)
        
        # 1. Evolve all islands independently in parallel
        evolved_pop, next_state = jax.vmap(self._island_loop, in_axes=(0, 0, 0, None))(
            multi_pop, multi_state, keys, global_gen_start
        )
        
        # 2. Share genetic material globally
        migrated_pop = self.migrate(k_migrate, evolved_pop)
        
        return migrated_pop, next_state
