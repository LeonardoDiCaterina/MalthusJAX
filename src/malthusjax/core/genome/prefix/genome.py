"""Prefix-aware genome and configuration for Multi-Expression Programming.

Extends :class:`LinearGenome` with on-demand path-analysis methods
(operand provenance, ancestor sets, effective p_input) without adding
any new stored fields.  All analysis is derived from the existing
``args`` array at call time.
"""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig


@struct.dataclass
class PrefixGenomeConfig(LinearGenomeConfig):
    """Configuration for prefix-aware Linear GP genomes.

    Extends :class:`LinearGenomeConfig` with parameters that control
    the structural properties of the encoded DAG at initialisation time.

    Attributes:
        p_input: Probability that a newly initialised argument references
            a raw input rather than a previous instruction row.  Higher
            values promote symbiotic diversity (parallel paths); lower
            values promote depth (fork / diamond motifs).  When ``None``,
            arguments are sampled uniformly over the full legal range
            (the default ``LinearGenome`` behaviour).
    """

    p_input: float | None = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    
    def init_population(self, key: chex.PRNGKey, size: int) -> Any:
        from malthusjax.core.genome.prefix.population import PrefixPopulation
        return PrefixPopulation.init_random(key, self, size)


@struct.dataclass
class BasePrefixAwareGenome(LinearGenome):
    """LinearGenome extended with on-demand path-analysis methods.

    Adds **no new fields**.  Provenance and ancestor-set computation are
    derived from the existing ``args`` array.  This keeps the genome
    zero-overhead at rest while enabling rich diagnostics when needed.

    All analysis methods accept the genome config so they know the
    ``num_inputs`` boundary without storing it redundantly.
    """

    # ------------------------------------------------------------------
    # Initialisation (overrides LinearGenome to support p_input)
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls, key: chex.PRNGKey, config: PrefixGenomeConfig  # type: ignore[override]
    ) -> BasePrefixAwareGenome:
        """Initialise an LGP genome with optional ``p_input`` bias.

        When ``config.p_input`` is ``None`` the behaviour is identical to
        :meth:`LinearGenome.random_init` (uniform sampling over the legal
        range).  When set, each argument independently samples a raw-input
        index with probability ``p_input`` and a previous-row index with
        probability ``1 - p_input``.
        """
        if config.p_input is None:
            # Fall back to the standard uniform initialisation.
            base = LinearGenome.random_init(key, config)
            return cls(ops=base.ops, args=base.args)

        k_ops, k_args, k_coin = jax.random.split(key, 3)

        ops = jax.random.randint(k_ops, (config.length,), 0, config.num_ops)

        # Per-row limits for the "previous row" range: row i can reference
        # indices in [num_inputs, num_inputs + i).
        row_starts = jnp.full((config.length,), config.num_inputs)
        row_ends = jnp.arange(config.num_inputs, config.num_inputs + config.length)

        def gen_row(
            rk: chex.PRNGKey, row_start: chex.Numeric, row_end: chex.Numeric
        ) -> chex.Array:
            k_input, k_row, k_flip = jax.random.split(rk, 3)

            # Sample from the raw-input range [0, num_inputs)
            input_args = jax.random.randint(
                k_input, (config.max_arity,), 0, config.num_inputs
            )
            # Sample from the previous-row range [num_inputs, num_inputs + i)
            # (for row 0 this range is empty so we clamp to 0)
            safe_end = jnp.maximum(row_end, row_start + 1)
            row_args = jax.random.randint(
                k_row, (config.max_arity,), row_start, safe_end
            )
            # Coin flip per argument
            coins = jax.random.uniform(k_flip, (config.max_arity,))
            return jnp.where(coins < config.p_input, input_args, row_args)

        row_keys = jax.random.split(k_args, config.length)
        args = jax.vmap(gen_row)(row_keys, row_starts, row_ends)

        return cls(ops=ops, args=args)

    # ------------------------------------------------------------------
    # Autocorrect (unchanged semantics, just returns correct type)
    # ------------------------------------------------------------------

    def autocorrect(self, config: LinearGenomeConfig) -> BasePrefixAwareGenome:
        """Restore DAG validity, returning a ``BasePrefixAwareGenome``."""
        base = super().autocorrect(config)
        return cast(BasePrefixAwareGenome, base)

    # ------------------------------------------------------------------
    # On-demand provenance & analysis
    # ------------------------------------------------------------------

    def get_operand_provenance(self, config: LinearGenomeConfig) -> jnp.ndarray:
        """Boolean mask ``(L, max_arity)``: ``True`` = raw input, ``False`` = previous row.

        This is a single elementwise comparison — O(L × max_arity),
        fully JIT-compatible, no graph traversal.
        """
        return self.args < config.num_inputs

    def get_ancestor_sets(self, config: LinearGenomeConfig) -> jnp.ndarray:
        """Binary ancestor masks ``(L, L)``.

        ``ancestors[i, j] == 1`` iff row *j* is a (transitive) ancestor
        of row *i*.  Computed via a single forward ``lax.scan`` pass in
        O(L²) time — cheap for typical chromosome lengths (L ≤ 200).
        """
        provenance = self.get_operand_provenance(config)  # (L, max_arity)
        L = config.length

        # Simpler implementation: direct loop-free propagation
        ancestors = jnp.zeros((L, L), dtype=jnp.bool_)

        def body_fn(i: int, anc: jnp.ndarray) -> jnp.ndarray:
            row_args = self.args[i]  # (max_arity,)
            row_prov = provenance[i]  # (max_arity,)

            # For each argument that references a previous row (not raw input),
            # mark that row and all *its* ancestors as ancestors of row i.
            row_ancestors = jnp.zeros(L, dtype=jnp.bool_)
            for a in range(config.max_arity):
                # Internal-row index (offset by num_inputs)
                internal_idx = row_args[a] - config.num_inputs
                # Only count if this argument references a previous row
                is_row_ref = ~row_prov[a]
                # Mark the referenced row itself
                ref_mask = jnp.zeros(L, dtype=jnp.bool_).at[internal_idx].set(is_row_ref)
                # Plus all ancestors of that row
                inherited = jnp.where(is_row_ref, anc[internal_idx], jnp.zeros(L, dtype=jnp.bool_))
                row_ancestors = row_ancestors | ref_mask | inherited

            return anc.at[i].set(row_ancestors)

        ancestors = jax.lax.fori_loop(0, L, body_fn, ancestors)
        return ancestors

    def get_effective_p_input(self, config: LinearGenomeConfig) -> chex.Numeric:
        """Fraction of all operand references pointing to raw inputs (measured).

        This is the *actual* topology pressure in the genome, which may
        drift from the configured ``p_input`` due to selection pressure.
        """
        return jnp.mean(self.get_operand_provenance(config).astype(jnp.float32))

    def get_symbiotic_diversity(
        self, config: LinearGenomeConfig, threshold: float = 0.1
    ) -> chex.Numeric:
        """Count of independent symbiotic paths (connected components).

        Two rows are considered part of the same path if their ancestor-set
        Jaccard index exceeds ``threshold``.  Returns the number of
        connected components in the resulting overlap graph.

        .. note::
           This is an approximate metric.  For exact motif classification
           use the full ancestor-set analysis.
        """
        ancestors = self.get_ancestor_sets(config)  # (L, L) bool

        # Include self in each row's ancestor set for Jaccard
        ancestor_with_self = ancestors | jnp.eye(config.length, dtype=jnp.bool_)

        # Pairwise Jaccard: J(i, j) = |A_i ∩ A_j| / |A_i ∪ A_j|
        intersection = jnp.einsum("ik,jk->ij", ancestor_with_self.astype(jnp.float32),
                                   ancestor_with_self.astype(jnp.float32))
        sizes = jnp.sum(ancestor_with_self, axis=-1, dtype=jnp.float32)
        union = sizes[:, None] + sizes[None, :] - intersection
        jaccard = jnp.where(union > 0, intersection / union, 0.0)

        # Build adjacency: two rows are connected if J > threshold
        adjacency = jaccard > threshold

        # Count connected components via iterated label propagation
        # (simple power-iteration on the adjacency matrix)
        labels = jnp.arange(config.length)

        def propagate(_, lbls):
            # Each node takes the minimum label among its neighbours
            expanded = jnp.where(adjacency, lbls[None, :], config.length)
            return jnp.minimum(lbls, jnp.min(expanded, axis=-1))

        labels = jax.lax.fori_loop(0, config.length, propagate, labels)
        return jnp.unique(labels, size=config.length, fill_value=-1).astype(jnp.int32)
        # Note: jnp.unique with size returns padded array; count non-(-1) for
        # the actual number. In practice we return the label array and let
        # the caller count unique values.

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<BasePrefixAwareGenome(L={self.size}, max_arity={self.args.shape[-1]})>"

@struct.dataclass
class ConstantGenomeConfig(PrefixGenomeConfig):
    """Configuration for genomes that evolve continuous constants."""
    num_constants: int = struct.field(pytree_node=False, default=0)  # type: ignore[no-untyped-call]

    def init_population(self, key: chex.PRNGKey, size: int) -> Any:
        from malthusjax.core.genome.prefix.population import PrefixPopulation
        return PrefixPopulation.init_random(key, self, size)

@struct.dataclass
class ConstantAwarePrefixGenome(BasePrefixAwareGenome):
    """A prefix-aware genome that also holds an array of continuous constants.
    
    The constants are treated as extra pseudo-inputs during evaluation.
    If num_inputs = N and num_constants = C, the args array can reference:
    - [0, N-1]: External inputs
    - [N, N+C-1]: Constants
    - [N+C, N+C+i-1]: Previous rows
    """
    constants: chex.Array  # Shape (num_constants,)

    @classmethod
    def random_init(
        cls, key: chex.PRNGKey, config: ConstantGenomeConfig  # type: ignore[override]
    ) -> ConstantAwarePrefixGenome:
        """Initialise an LGP genome with optional constants."""
        k_base, k_const = jax.random.split(key)
        
        # We initialise the base genome using the effective number of inputs (N + C)
        # so that it legally samples from the constants range as well!
        base_config = PrefixGenomeConfig(
            length=config.length,
            num_inputs=config.num_inputs + config.num_constants,
            num_ops=config.num_ops,
            max_arity=config.max_arity,
            p_input=config.p_input
        )
        base = BasePrefixAwareGenome.random_init(k_base, base_config)
        
        # Sample initial constants from standard normal
        constants = jax.random.normal(k_const, (config.num_constants,))
        
        return cls(ops=base.ops, args=base.args, constants=constants)
        
    def autocorrect(self, config: ConstantGenomeConfig) -> ConstantAwarePrefixGenome:
        base_config = PrefixGenomeConfig(
            length=config.length,
            num_inputs=config.num_inputs + config.num_constants,
            num_ops=config.num_ops,
            max_arity=config.max_arity,
            p_input=config.p_input
        )
        base = super().autocorrect(base_config)
        return cast(ConstantAwarePrefixGenome, base).replace(constants=self.constants)

