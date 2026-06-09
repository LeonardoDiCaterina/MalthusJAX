import chex
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple, Optional
from malthusjax.operators.base import BaseSelection, C, P

@struct.dataclass
class EvoSaxMimicSelection(BaseSelection[P, C]):
    elite_k: int = struct.field(pytree_node=False, default=10)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any) -> chex.Array:
        rng = keys if keys.ndim <= 1 else keys[0]
        pop_size = fitness.shape[0]
        idx = jnp.argsort(fitness)
        p = (jnp.arange(pop_size) < self.elite_k).astype(jnp.float32)
        p_norm = p / jnp.sum(p)
        return jax.random.choice(rng, idx, shape=(self.num_selections,), replace=True, p=p_norm)

    def __call__(self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any) -> Tuple[chex.Array, chex.Array]:
        fitness = jnp.asarray(getattr(population, "fitness", population))
        rng = keys if keys.ndim <= 1 else keys[0]
        
        # Select parents
        parent_idx = self._select(rng, fitness)
        
        # Elites to preserve (for SimpleGA, this matches n_elites)
        idx = jnp.argsort(fitness)
        elite_idx = idx[:self.n_elites]
        
        return parent_idx, elite_idx
