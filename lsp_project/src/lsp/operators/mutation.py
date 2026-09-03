"""MEP Mutation Operators.

Implements the two mutation variants described in Oltean & Dumitrescu (2002):

    - Standard mutation   (§3.7.2.1)
    - Smooth mutation     (§3.7.2.2)

Plus ``AnnealedTopologicalMutation`` — a structured variant that anneals the
probability that new argument pointers reference *input terminals* vs
*intermediate results*, used for the differentiable MO workflow.

JAX exception handling (§3.8):
    The paper mutates genes that raise exceptions (e.g. div-by-zero) into
    terminals.  In JAX, all functions are numerically protected (via
    ``jnp.where`` guards and ``jnp.nan_to_num`` in the interpreter) so no
    runtime exceptions occur.  This matches the paper's intent — infertile
    individuals are never produced.
"""

from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
from malthusjax.operators.base import BaseMutation

# ---------------------------------------------------------------------------
# Helper: sample row-limited arg indices
# ---------------------------------------------------------------------------


def _sample_args(
    key: chex.PRNGKey,
    config: LinearGenomeConfig,
) -> chex.Array:
    """Sample (L, max_arity) arg indices satisfying per-row DAG bounds.

    For row i, valid arg indices are [0, num_inputs + i).
    We sample from [0, num_inputs + L) then clip per-row.
    """
    L = config.length
    N = config.num_inputs
    raw = jax.random.randint(key, (L, config.max_arity), 0, N + L)
    # Per-row upper bound: num_inputs + row_index (exclusive)
    row_limits = jnp.arange(N, N + L)  # (L,)
    return jnp.clip(raw, 0, row_limits[:, None] - 1)


# ---------------------------------------------------------------------------
# Standard mutation  (§3.7.2.1)
# ---------------------------------------------------------------------------


from malthusjax.composer.decorators import register_mutation

@register_mutation(name="mep_micro")
@struct.dataclass
class MEPMicroMutation(BaseMutation[LinearGenome, LinearGenomeConfig]):
    """Standard MEP mutation — change any gene (op + args) with probability p_m.

    For each gene i (independently):
        - With probability ``mutation_rate``: replace the entire gene with a
          freshly sampled opcode and arg pointers (DAG-valid).
        - Otherwise: keep the gene unchanged.

    This is equivalent to Oltean's "standard mutation" (§3.7.2.1) where a
    terminal can become a function or vice-versa, with new random pointers.

    Args:
        mutation_rate: Per-gene mutation probability (p_m in the paper;
            suggested value for L=7 ≈ 0.28, i.e. ≈2 mutations per chromosome).
    """

    mutation_rate: float = struct.field(pytree_node=False, default=0.28)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # key 0: should_mutate mask  (L booleans)
        # key 1: new opcodes         (L integers)
        # key 2: new arg pointers    (L × max_arity integers)
        return 3

    def _generate_noise(
        self,
        keys: chex.Array,
        config: LinearGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Tier 2 — generate mutation mask, new opcodes, new arg indices."""
        L = config.length
        should_mutate = jax.random.bernoulli(keys[0], p=self.mutation_rate, shape=(L,))
        new_ops = jax.random.randint(keys[1], (L,), 0, config.num_ops)
        new_args = _sample_args(keys[2], config)
        return should_mutate, new_ops, new_args

    def _mutate_one(
        self,
        genome: LinearGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — apply mutation mask."""
        should_mutate, new_ops, new_args = noise_data  # (L,), (L,), (L, max_arity)

        mutated_ops = jnp.where(should_mutate, new_ops, genome.ops)
        mutated_args = jnp.where(should_mutate[:, None], new_args, genome.args)

        result = genome.replace(ops=mutated_ops, args=mutated_args)
        # autocorrect clips any out-of-range indices — should be a no-op here
        return result.autocorrect(config)


# ---------------------------------------------------------------------------
# Smooth mutation  (§3.7.2.2)
# ---------------------------------------------------------------------------


@struct.dataclass
class SmoothMutation(BaseMutation[LinearGenome, LinearGenomeConfig]):
    """Smooth MEP mutation — per-symbol perturbation within each gene.

    Unlike ``MEPMicroMutation`` which replaces entire genes, smooth mutation
    independently perturbs each symbol (opcode or individual arg pointer)
    with a fixed probability ``smooth_rate``.

    Suggested value for 2-argument functions: ``smooth_rate = 0.33``
    (one expected mutation per gene on average, matching "one position
    mutation per gene" in the paper §3.7.2.2).

    Args:
        smooth_rate: Per-symbol mutation probability (p_sm in the paper).
    """

    smooth_rate: float = struct.field(pytree_node=False, default=0.33)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # key 0: op mutation mask    (L booleans)
        # key 1: new opcodes
        # key 2: arg mutation mask   (L × max_arity booleans)
        # key 3: new arg pointers
        return 4

    def _generate_noise(
        self,
        keys: chex.Array,
        config: LinearGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Tier 2 — per-symbol masks + replacement values."""
        L = config.length
        mut_op_mask = jax.random.bernoulli(keys[0], p=self.smooth_rate, shape=(L,))
        new_ops = jax.random.randint(keys[1], (L,), 0, config.num_ops)
        mut_arg_mask = jax.random.bernoulli(
            keys[2], p=self.smooth_rate, shape=(L, config.max_arity)
        )
        new_args = _sample_args(keys[3], config)
        return mut_op_mask, new_ops, mut_arg_mask, new_args

    def _mutate_one(
        self,
        genome: LinearGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — independently replace masked opcodes and arg pointers."""
        mut_op_mask, new_ops, mut_arg_mask, new_args = noise_data

        mutated_ops = jnp.where(mut_op_mask, new_ops, genome.ops)
        mutated_args = jnp.where(mut_arg_mask, new_args, genome.args)

        result = genome.replace(ops=mutated_ops, args=mutated_args)
        return result.autocorrect(config)


# ---------------------------------------------------------------------------
# Annealed Topological Mutation (MO / differentiable workflow)
# ---------------------------------------------------------------------------


@struct.dataclass
class AnnealedTopologicalMutation(BaseMutation[LinearGenome, LinearGenomeConfig]):
    """Annealed structural mutation with input-bias scheduling.

    Extends standard mutation with an annealed probability that new arg
    pointers reference *input terminals* (indices [0, num_inputs)) rather
    than *intermediate results* (indices [num_inputs, num_inputs+i)).

    This encourages broad exploration of terminal-heavy expressions early in
    the run (high ``p_input_start``), then shifts toward deeper intermediate
    combinations as training progresses (low ``p_input_end``).

    Args:
        op_rate:        Per-gene opcode mutation probability.
        arg_rate:       Per-(gene, arity) arg pointer mutation probability.
        p_input_start:  Input-bias at generation 0.
        p_input_end:    Input-bias at ``max_generations``.
    """

    op_rate: float = struct.field(pytree_node=False, default=0.1)
    arg_rate: float = struct.field(pytree_node=False, default=0.2)
    p_input_start: float = struct.field(pytree_node=False, default=0.1)
    p_input_end: float = struct.field(pytree_node=False, default=0.9)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # key 0: op mutation mask
        # key 1: new opcodes
        # key 2: arg mutation mask   (L × max_arity)
        # key 3: new arg pointers
        return 4

    def _generate_noise(
        self,
        keys: chex.Array,
        config: LinearGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Tier 2 — masks + annealed arg sampling."""
        L = config.length
        N = config.num_inputs
        max_gens = max(self.max_generations, 1)

        # Linear annealing of input-bias probability
        frac = jnp.clip(generation / max_gens, 0.0, 1.0)
        p_input = self.p_input_start + frac * (self.p_input_end - self.p_input_start)

        # Op mutation
        op_mask = jax.random.bernoulli(keys[0], p=self.op_rate, shape=(L,))
        new_ops = jax.random.randint(keys[1], (L,), 0, config.num_ops)

        # Arg mutation with annealed input-bias
        arg_mask = jax.random.bernoulli(keys[2], p=self.arg_rate, shape=(L, config.max_arity))

        # For each slot: with probability p_input → sample from [0, N)
        #                with probability 1-p_input → sample from [0, N+L) clipped per row
        use_input = jax.random.bernoulli(keys[3], p=p_input, shape=(L, config.max_arity))

        raw_full = jax.random.randint(keys[3], (L, config.max_arity), 0, N + L)
        row_limits = jnp.arange(N, N + L)
        full_args = jnp.clip(raw_full, 0, row_limits[:, None] - 1)

        input_args = jax.random.randint(keys[3], (L, config.max_arity), 0, jnp.maximum(N, 1))

        new_args = jnp.where(use_input, input_args, full_args)

        return op_mask, new_ops, arg_mask, new_args

    def _mutate_one(
        self,
        genome: LinearGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — apply op and arg mutation masks."""
        op_mask, new_ops, arg_mask, new_args = noise_data

        mutated_ops = jnp.where(op_mask, new_ops, genome.ops)
        mutated_args = jnp.where(arg_mask, new_args, genome.args)

        result = genome.replace(ops=mutated_ops, args=mutated_args)
        return result.autocorrect(config)


__all__ = [
    "MEPMicroMutation",
    "SmoothMutation",
    "AnnealedTopologicalMutation",
]
