"""LSP Genome — PrefixGenomeConfig and BasePrefixAwareGenome.

Thin layer on top of malthusjax.core.genome.linear_genome.LinearGenome
providing LSP-project naming conventions and ancestor-set analysis for
parsimony pressure (active node count) in multi-objective evaluators.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig

# ---------------------------------------------------------------------------
# Config alias
# ---------------------------------------------------------------------------


@struct.dataclass
class PrefixGenomeConfig(LinearGenomeConfig):
    """Configuration for Prefix Aware Genomes.

    Adds `init_population` to comply with the GeneticEngine interface.
    """

    @property
    def dtype(self) -> Any:
        return jnp.int32

    def init_population(self, key: chex.PRNGKey, size: int) -> "BasePopulation":
        from malthusjax.core.base import BasePopulation

        keys = jax.random.split(key, size)

        # We need to construct genomes. BasePrefixAwareGenome is what we use.
        genomes = jax.vmap(BasePrefixAwareGenome.random_init, in_axes=(0, None))(keys, self)

        return BasePopulation(genes=genomes, fitness=jnp.zeros(size), info={})


# ---------------------------------------------------------------------------
# BasePrefixAwareGenome
# ---------------------------------------------------------------------------


@struct.dataclass
class BasePrefixAwareGenome(LinearGenome):
    """LinearGenome extended with ancestor-set analysis.

    Adds ``get_ancestor_sets``, which computes the transitive closure of the
    argument-pointer DAG.  This is used by ``DifferentiableMOEvaluator`` to
    count the number of *active* (reachable) nodes for parsimony pressure.

    No new PyTree fields are added — this class is structurally identical to
    ``LinearGenome`` and can be constructed from any ``LinearGenome`` via::

        BasePrefixAwareGenome(ops=genome.ops, args=genome.args)
    """

    def get_ancestor_sets(self, config: PrefixGenomeConfig) -> chex.Array:
        """Compute (L, L) boolean ancestor matrix via ``jax.lax.scan``.

        ``ancestors[i, j] = True`` iff instruction-row *j* is an ancestor
        (direct or transitive input) of instruction-row *i*.

        Uses the DAG ordering guarantee (``args[i] < num_inputs + i``) so a
        single top-down scan suffices — no fixed-point iteration required.

        Args:
            config: Genome configuration providing ``length``, ``num_inputs``,
                and ``max_arity``.

        Returns:
            Boolean array of shape ``(L, L)``.
        """
        L: int = config.length
        N: int = config.num_inputs
        max_arity: int = config.max_arity

        def _step(
            ancestors: chex.Array, row_data: tuple[chex.Array, chex.Array]
        ) -> tuple[chex.Array, None]:
            i, args_row = row_data  # i: scalar int, args_row: (max_arity,)

            is_intermediate = args_row >= N  # (max_arity,)
            idxs = jnp.clip(args_row - N, 0, L - 1)  # row indices into ancestors

            # Direct parents: for each arg that points to an intermediate row,
            # mark that row as a direct ancestor of row i.
            direct = jnp.zeros(L, dtype=jnp.bool_)
            transitive = jnp.zeros(L, dtype=jnp.bool_)

            # Unrolled over max_arity (static at trace time).
            for k in range(max_arity):
                idx_k = idxs[k]
                is_int_k = is_intermediate[k]
                direct = direct.at[idx_k].set(direct[idx_k] | is_int_k)
                transitive = transitive | (ancestors[idx_k] & is_int_k)

            new_row = direct | transitive
            new_ancestors = ancestors.at[i].set(new_row)
            return new_ancestors, None

        init = jnp.zeros((L, L), dtype=jnp.bool_)
        xs = (jnp.arange(L), self.args)  # parallel scan inputs
        final_ancestors, _ = jax.lax.scan(_step, init, xs)
        return final_ancestors


__all__ = ["PrefixGenomeConfig", "BasePrefixAwareGenome"]
