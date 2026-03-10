from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P, _field


@struct.dataclass
class RouletteSelection(BaseSelection[P, C]):
    """
    Fitness-Proportional Roulette Selection (Adaptive Paths).
    Strategy: Sample indices with probability ∝ fitness via softmax(fitness/temperature).
    Shape contract: fitness (pop_size,) → selected_indices (num_selections,).
    Key budget: 1 pre-allocated subkey (uniform noise or choice sampling).
    Temperature: Controls selection pressure (low→exploitation, high→uniform).
    Branching: Gumbel-Max for num_selections==pop_size (O(1) parallel), Categorical otherwise.
    Trade-off: Gumbel-Max (fast, O(N*M) memory), Categorical (memory-efficient, softmax stable).
    Use when: Fitness landscape well-characterized; need fitness-weighted exploration.
    """

    temperature: float = _field(pytree_node=False, default=1.0)
    use_gumbel_trick: bool = _field(pytree_node=False, default=True)
    chunk_size: int = _field(pytree_node=False, default=1024)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """
        Samples num_selections parents with probability ∝ exp(fitness/temperature).
        Returns: (num_selections,) indices into [0, pop_size).
        Uses Gumbel-Max trick when num_selections==pop_size (efficient parallel path);
        falls back to softmax+jax.random.choice for other configurations.
        """
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
        pop_size = fitness.shape[0]
        logits = fitness / self.temperature
        if self.use_gumbel_trick and self.num_selections == pop_size:
            if pop_size <= self.chunk_size:
                uniform_noise = jax.random.uniform(rng, shape=(self.num_selections, pop_size))
                gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
                return jnp.argmax(logits + gumbel_noise, axis=1)
            else:
                num_chunks = (self.num_selections + self.chunk_size - 1) // self.chunk_size

                def _chunk_body(
                    carry: chex.Array, _: Any
                ) -> tuple[chex.Array, chex.Array]:
                    key = carry
                    key, subkey = jax.random.split(key)
                    uniform = jax.random.uniform(
                        subkey, shape=(self.chunk_size, pop_size)
                    )
                    gumbel = -jnp.log(-jnp.log(uniform))
                    indices = jnp.argmax(logits + gumbel, axis=1)
                    return key, indices

                _, all_indices = jax.lax.scan(
                    _chunk_body, rng, None, length=num_chunks
                )
                return all_indices.reshape(-1)[: self.num_selections]
        else:
            probs = jax.nn.softmax(logits)
            return jax.random.choice(
                rng, a=pop_size, shape=(self.num_selections,), p=probs, replace=True
            )
