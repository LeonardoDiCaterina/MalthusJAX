from abc import abstractmethod
from numbers import Integral
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

    Local evolution and migration are JAX-traceable and fused by the adapter.

    This wrapper seamlessly upgrades any 1D `BaseEngine` into a 2D Island Model by
    wrapping the engine's init and step methods in `jax.vmap` across the `num_islands` axis,
    and then applying a customizable topological migration policy.
    """

    engine: E
    num_islands: int = struct.field(pytree_node=False)
    migration_interval: int = struct.field(pytree_node=False)
    num_migrants: int = struct.field(pytree_node=False)

    def __post_init__(self):
        for name, lower in (("num_islands", 1), ("migration_interval", 1), ("num_migrants", 0)):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")

    def evolve_one(self, multi_state):
        return jax.vmap(self.engine.step)(multi_state)

    def migrate_state(self, evolved_state):
        if self.num_migrants == 0:
            return evolved_state
        migration_key, new_key = jax.random.split(evolved_state.rng_key[0])
        population = self.migrate(migration_key, evolved_state.population)
        return evolved_state.replace(population=population,
            rng_key=evolved_state.rng_key.at[0].set(new_key))

    def init_state(self, key: chex.PRNGKey) -> Any:
        """Initializes the engine perfectly mapped across independent islands."""
        keys = jax.random.split(key, self.num_islands)
        # multi_state contains the 2D population
        multi_state = jax.vmap(self.engine.init_state, in_axes=(0,))(keys)
        if self.num_migrants > multi_state.population.fitness.shape[1]:
            raise ValueError("num_migrants exceeds the per-island population")
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

        return self.migrate_state(evolved_state), history
