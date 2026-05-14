from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P, _field


@struct.dataclass
class RouletteSelection(BaseSelection[P, C]):
    """Fitness-Proportional Roulette Selection (Adaptive Paths).

    In roulette selection, individuals are selected with probability proportional
    to their fitness. A softmax transformation converts fitness values to a valid
    probability distribution, with temperature controlling selection pressure.

    This approach allows the algorithm to adaptively adjust selection pressure based
    on fitness diversity across generations.

    **String Specification Format**::

        "roulette:num_selections=INT[,temperature=FLOAT]"

    Examples::

        "roulette"  # Defaults (num_selections=4, temperature=1.0)
        "roulette:num_selections=25"  # Custom num_selections, default temperature
        "roulette:num_selections=50,temperature=0.5"  # Both custom (lower temp = more pressure)
        "roulette:temperature=2.0"  # Custom temperature, default num_selections

    Parameters
    ----------
    temperature : float, optional
        Controls selection pressure in the softmax transformation.
        Valid range: (0, ∞)
        - temperature=0.1: Very strong exploitation (high-fitness individuals dominate)
        - temperature=1.0: Balanced (fitness-proportional, default)
        - temperature=5.0: Weak selection, more uniform (higher diversity)
        Default: 1.0 (balanced fitness-proportional selection).

    use_gumbel_trick : bool, optional
        If True (default), uses the Gumbel-Max trick for fast parallel sampling
        when ``num_selections == population_size``. Otherwise uses categorical sampling.
        Default: True (faster for full replacement).

    chunk_size : int, optional
        When using Gumbel-Max with populations larger than this, splits into chunks
        to control memory usage. Default: 1024.

    num_selections : int
        Number of individuals to select. Set by the engine.

    Notes
    -----
    **⚠️  CRITICAL FITNESS REQUIREMENT**: Roulette selection requires **non-negative
    fitness values**. If your fitness function can return negative values (e.g.,
    minimization problems with negative scores), you must either:

    1. **Shift fitness**: Add a constant to all fitness values to make them positive
    2. **Transform fitness**: Use ``1.0 / (1.0 + loss)`` or similar
    3. **Use tournament selection instead**: Recommended if you're unsure about fitness range

    Violating this requirement will cause NaN values in the softmax and invalid
    selection behavior.

    **Temperature Tuning**:
    - Low temperature (0.1-0.5): Aggressive selection pressure, risk of premature
      convergence to local optima
    - Medium temperature (1.0-2.0): Balanced, recommended for exploiting good regions
      while maintaining diversity
    - High temperature (5.0+): Uniform sampling, high diversity, slow convergence

    **Computational Complexity**:
    - With Gumbel-Max and num_selections == pop_size: O(N log N) due to reduction
    - Otherwise: O(N + num_selections) via categorical sampling
    - Memory: O(N × num_selections) for Gumbel-Max, O(N) for categorical

    **When to Use**:
    - Problem has well-characterized fitness landscape
    - You want fitness-weighted selection (high-fitness individuals breed more)
    - Diversity is maintained through other mechanisms (mutation, crossover)
    - All fitness values are guaranteed non-negative

    **Recommendation**: If you're unsure about fitness ranges or want robustness,
    use :class:`TournamentSelection` instead (works with any fitness function).
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
        """Draw parents proportional to fitness using a softmax or Gumbel-Max.

        In minimization convention (lower=better), fitness values are negated to create
        a proper probability distribution where lower original fitness gets higher selection probability.
        When ``num_selections == pop_size`` and the Gumbel trick is enabled,
        a parallel variant computes all samples in one shot; otherwise a
        categorical draw is performed.
        """
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
        pop_size = fitness.shape[0]
        # Negate fitness for minimization: lower values become higher (better)
        logits = -fitness / self.temperature
        if self.use_gumbel_trick and self.num_selections == pop_size:
            if pop_size <= self.chunk_size:
                uniform_noise = jax.random.uniform(rng, shape=(self.num_selections, pop_size))
                gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
                return jnp.argmax(logits + gumbel_noise, axis=1)
            else:
                num_chunks = (self.num_selections + self.chunk_size - 1) // self.chunk_size

                def _chunk_body(carry: chex.Array, _: Any) -> tuple[chex.Array, chex.Array]:
                    key = carry
                    key, subkey = jax.random.split(key)
                    uniform = jax.random.uniform(subkey, shape=(self.chunk_size, pop_size))
                    gumbel = -jnp.log(-jnp.log(uniform))
                    indices = jnp.argmax(logits + gumbel, axis=1)
                    return key, indices

                _, all_indices = jax.lax.scan(_chunk_body, rng, None, length=num_chunks)
                return all_indices.reshape(-1)[: self.num_selections]
        else:
            probs = jax.nn.softmax(logits)
            return jax.random.choice(
                rng, a=pop_size, shape=(self.num_selections,), p=probs, replace=True
            )
