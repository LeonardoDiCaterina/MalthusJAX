"""
Selection Operators.
Refactored for the 'Consumer Paradigm': Pure index generation.
"""

from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

_field: Any = struct.field


@struct.dataclass
class ElitePoolSelection(BaseSelection[P, C]):
    """Elite Pool Selection (High Performance).

    In elite pool selection, the algorithm:
    1. Identifies the top ``elite_k`` individuals by fitness
    2. For each selection, uniformly randomly picks from this elite pool
    3. Guarantees only the best genes are preserved and propagated

    This provides maximum exploitation of high-fitness solutions but with limited
    diversity (all selected parents come from a small elite pool).

    **String Specification Format**::

        "elite_pool:num_selections=INT[,elite_k=INT]"

    Examples::

        "elite_pool"  # Defaults (num_selections=4, elite_k=2)
        "elite_pool:num_selections=50"  # Custom num_selections, default elite_k
        "elite_pool:num_selections=50,elite_k=10"  # Both custom
        "elite_pool:elite_k=5"  # Custom elite pool size, default num_selections

    Parameters
    ----------
    elite_k : int, optional
        Number of top individuals to include in the selection pool.
        Valid range: 1 to population_size.
        - elite_k=1: Only the single best individual is selected (very aggressive)
        - elite_k=3-5: Small elite pool (typical for fine-tuning)
        - elite_k = pop_size // 3: Moderate elite pool (1/3 of population)
        - elite_k = pop_size // 2: Large elite pool (more diversity)
        Default: 10.

    num_selections : int
        Number of individuals to select from the elite pool.
        Set by the engine during :meth:`set_input_length`.
        Note: All selections come from the ``elite_k`` best individuals.

    Notes
    -----
    **Fitness Requirements**: Elite pool selection supports all fitness ranges
    (positive, negative, zero, or mixed). The algorithm ranks individuals and
    selects from the top-k, making it robust to any fitness function.

    **Selection Pressure**: Very high—only the top ``elite_k`` individuals ever
    contribute genes to the next generation. This maximizes exploitation but
    minimizes diversity.

    **Design Trade-offs**:
    - **Pros**: Fast convergence, preserves best genes, O(N) complexity
    - **Cons**: Very low diversity, risk of premature convergence,
      limited exploration of search space

    **When to Use**:
    1. **Final refinement phase**: Late generations when you're optimizing near
       a known good region
    2. **High-dimensional problems with clear optima**: Problems where top-1%
       solutions are significantly better than others
    3. **Hybrid approaches**: Combine with tournament or roulette selection
       (e.g., 30% elite pool selections, 70% tournament selections)
    4. **Small elite pool (k ≤ 5)**: When you trust your problem has a clear gradient

    **When NOT to Use**:
    - Problem has many equally-good optima (high confusion)
    - Early exploration phase is critical (use tournament instead)
    - Diversity maintenance is important (use tournament or roulette)
    - Problem landscape is multimodal (you may need to explore more)

    **Computational Complexity**:
    - Filtering elite: O(N) via ``jnp.argpartition`` (very fast)
    - Sampling from pool: O(num_selections) with uniform random selection
    - **Total: O(N + num_selections)** (efficient)

    **Recommendation**: Start with :class:`TournamentSelection` (tournament_size=3)
    for general problems. If convergence is too slow near the optimum, introduce
    a small elite pool (5-10% of population) mixed with tournament selection.
    """

    elite_k: int = _field(pytree_node=False, default=10)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Choose parents uniformly from the top *elite_k* individuals.

        This simpler primitive method is mainly used when the fused
        ``__call__`` path (which also extracts elites) is not required.
        """
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]

        pop_size = fitness.shape[0]
        pool_k = min(self.elite_k, pop_size)

        if pool_k >= pop_size:
            best_k_indices = jnp.arange(pop_size)
        else:
            best_k_indices = jnp.argsort(fitness)[:pool_k]

        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=pool_k
        )
        return best_k_indices[random_selections]

    def __call__(
        self,
        keys: chex.Array,
        population: P,
        config: Optional[C] = None,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        """Select parents from the elite pool and simultaneously identify elites.

        This implementation avoids a second O(N) scan by performing a single
        ``argpartition`` on the combined effect of ``elite_k`` and ``n_elites``.
        """
        fitness: chex.Array = jnp.asarray(getattr(population, "fitness", population))

        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]

        pop_size = fitness.shape[0]
        k = min(max(self.elite_k, self.n_elites), pop_size)

        if k >= pop_size:
            top_k_idx = jnp.arange(pop_size)
        else:
            top_k_idx = jnp.argsort(fitness)[:k]

        pool_k = min(self.elite_k, pop_size)

        if self.n_elites == 0:
            pool = top_k_idx[:pool_k]
            elite_idx = jnp.zeros(0, dtype=jnp.int32)
        elif self.elite_k == self.n_elites:
            pool = top_k_idx[:pool_k]
            elite_idx = top_k_idx[: self.n_elites]
        else:
            sorted_within = jnp.argsort(-fitness[top_k_idx])
            sorted_top_k = top_k_idx[sorted_within]
            pool = sorted_top_k[:pool_k]
            elite_idx = sorted_top_k[: self.n_elites]

        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=pool_k
        )
        parent_idx = pool[random_selections]

        return parent_idx, elite_idx
