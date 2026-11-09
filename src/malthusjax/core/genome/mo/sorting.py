"""Pure-JAX Sorting Primitives for NSGA-II."""

import chex
import jax
import jax.numpy as jnp


def compute_dominance_matrix(fitness: chex.Array, maximize: bool = False) -> chex.Array:
    """Computes the (N, N) boolean dominance matrix.

    Args:
        fitness: (N, M) array of multi-objective fitnesses.
        maximize: If True, higher fitness is better.

    Returns:
        D: (N, N) boolean array where D[i, j] is True if individual i dominates j.
    """
    f_diff = fitness[:, None, :] - fitness[None, :, :]

    if maximize:
        no_worse = jnp.all(f_diff >= 0, axis=-1)
        strictly_better = jnp.any(f_diff > 0, axis=-1)
    else:
        no_worse = jnp.all(f_diff <= 0, axis=-1)
        strictly_better = jnp.any(f_diff < 0, axis=-1)

    return no_worse & strictly_better


def compute_pareto_ranks(dominance_matrix: chex.Array) -> chex.Array:
    """Computes the discrete Pareto rank for each individual using a fixed-iteration loop.

    This function avoids JAX dynamic shape errors by unrolling the front-peeling
    logic up to N times, which is the theoretical maximum number of fronts.

    Args:
        dominance_matrix: (N, N) boolean array where D[i, j] means i dominates j.

    Returns:
        ranks: (N,) integer array of Pareto ranks (0 is the best, i.e. non-dominated).
    """
    n = dominance_matrix.shape[0]

    def body_fn(i: int, ranks: chex.Array) -> chex.Array:
        unranked_mask = ranks == -1
        active_dominance = dominance_matrix & unranked_mask[:, None]
        is_dominated = jnp.any(active_dominance, axis=0)
        front_mask = unranked_mask & (~is_dominated)
        return jnp.where(front_mask, i, ranks)

    initial_ranks = jnp.full((n,), -1, dtype=jnp.int32)
    return jax.lax.fori_loop(0, n, body_fn, initial_ranks)


def compute_crowding_distance(fitness: chex.Array, ranks: chex.Array) -> chex.Array:
    """Front-local NSGA-II crowding, summed across objectives (not averaged).

    Constant objectives contribute zero, including their arbitrary sorted ends.
    Fronts with at most two members receive infinity. Ties use input-index order.
    Empty (0, M) input returns an empty vector. Shape/type mistakes raise at
    trace time; nonfinite fitness or negative ranks return all-NaN distances in
    eager and JIT execution, making invalid input explicit without callbacks.
    """
    if fitness.ndim != 2 or fitness.shape[1] == 0:
        raise ValueError("fitness must have shape (N, M) with at least one objective")
    if ranks.ndim != 1 or ranks.shape[0] != fitness.shape[0]:
        raise ValueError("ranks must have shape (N,) matching fitness")
    if not jnp.issubdtype(ranks.dtype, jnp.integer):
        raise ValueError("ranks must have integer dtype")
    if not (
        jnp.issubdtype(fitness.dtype, jnp.floating) or jnp.issubdtype(fitness.dtype, jnp.integer)
    ):
        raise ValueError("fitness must contain real numeric values")
    fitness = fitness.astype(jnp.result_type(fitness.dtype, jnp.float32))
    n, num_objectives = fitness.shape
    distances = jnp.zeros(n, dtype=fitness.dtype)
    if n == 0:
        return distances
    valid = jnp.all(jnp.isfinite(fitness)) & jnp.all(ranks >= 0)
    positions = jnp.arange(n)
    for m in range(num_objectives):
        idx = jnp.lexsort((positions, fitness[:, m], ranks))
        sorted_f = fitness[idx, m]
        sorted_r = ranks[idx]
        left = jnp.concatenate((jnp.array([True]), sorted_r[1:] != sorted_r[:-1]))
        right = jnp.concatenate((sorted_r[:-1] != sorted_r[1:], jnp.array([True])))
        # Forward/backward scans locate each member's own front, even for
        # noncontiguous rank labels. No dense N-by-N crowding mask is needed.
        start = jax.lax.associative_scan(jnp.maximum, jnp.where(left, positions, 0))
        end = jax.lax.associative_scan(
            jnp.minimum, jnp.where(right, positions, n - 1), reverse=True
        )
        low, high = sorted_f[start], sorted_f[end]
        span = high - low
        previous, following = jnp.roll(sorted_f, 1), jnp.roll(sorted_f, -1)
        ordinary = (following - previous) / jnp.where(span > 0, span, 1)
        # Finite endpoints can overflow their subtraction. Scale only in that
        # case; no epsilon floor changes distances on small nonzero ranges.
        scale = jnp.maximum(jnp.abs(low), jnp.abs(high))
        scale = jnp.where(scale > 0, scale, 1)
        scaled_span = high / scale - low / scale
        scaled = (following / scale - previous / scale) / jnp.where(scaled_span > 0, scaled_span, 1)
        gap = jnp.where(jnp.isfinite(span), ordinary, scaled)
        updates = jnp.where(span > 0, jnp.where(left | right, jnp.inf, gap), 0.0)
        updates = jnp.where(end - start < 2, jnp.inf, updates)
        distances = distances + jnp.zeros_like(distances).at[idx].set(updates)
    return jnp.where(valid, distances, jnp.nan)
