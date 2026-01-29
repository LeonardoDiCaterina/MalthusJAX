from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

# Internal alias to bypass mypy --strict errors on Flax fields
_field: Any = struct.field

@struct.dataclass
class RouletteSelection(BaseSelection[P, C]):
    """
    Selection operator that samples parents proportional to their fitness.
    Patterns:
    - Gumbel-Max: Fast O(1) parallel path for smaller populations.
    - Categorical: Memory-efficient O(N) path for large-scale evolution.
    """
    temperature: float = _field(pytree_node=False, default=1.0)

    # NEW: Toggle for memory safety on high populations
    use_gumbel_trick: bool = _field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self,
        keys: chex.Array,
        fitness: chex.Array,
        config: Optional[C] = None,
        **kwargs: Any
    ) -> chex.Array:
        """
        Samples indices proportional to fitness.
        Signature matched to BaseSelection to fix integration TypeErrors.
        """
        # Normalize key handling for both raw and engine-sliced keys
        rng = keys[0] if keys.ndim > 1 else keys
        pop_size = fitness.shape[0]

        # 1. Compute Logits with Numerical Stability
        # We use a standard shift to avoid overflow
        logits = fitness / self.temperature

        # 2. Logic Branching
        # Gumbel-Max is faster but uses O(num_selections * pop_size) memory
        if self.use_gumbel_trick and self.num_selections == pop_size:
            # === OPTIMIZED PATH: Gumbel-Max Trick ===
            # Best for N < 4096 to maximize GPU utilization
            uniform_noise = jax.random.uniform(rng, shape=(self.num_selections, pop_size))
            gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
            return jnp.argmax(logits + gumbel_noise, axis=1)

        else:
            # === MEMORY-EFFICIENT PATH ===
            # Standard path for high N or when trick is disabled.
            # Uses jax.nn.softmax for internal stability
            probs = jax.nn.softmax(logits)
            return jax.random.choice(
                rng, a=pop_size, shape=(self.num_selections,), p=probs, replace=True
            )
