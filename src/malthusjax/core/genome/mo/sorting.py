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
    """Computes the crowding distance for each individual within its Pareto front.
    
    Args:
        fitness: (N, M) array of multi-objective fitness values.
        ranks: (N,) array of integer Pareto ranks (0 is the best front).
    
    Returns:
        distances: (N,) array of crowding distances (higher is more diverse, jnp.inf for boundaries).
    """
    n, num_objectives = fitness.shape
    
    f_min = jnp.min(fitness, axis=0)
    f_max = jnp.max(fitness, axis=0)
    f_range = jnp.maximum(f_max - f_min, 1e-6)
    
    distances = jnp.zeros(n)
    
    for m in range(num_objectives):
        idx = jnp.lexsort((fitness[:, m], ranks))
        
        sorted_f = fitness[idx, m]
        sorted_r = ranks[idx]
        
        diffs = sorted_f[2:] - sorted_f[:-2]
        same_rank = sorted_r[2:] == sorted_r[:-2]
        norm_diffs = jnp.where(same_rank, diffs / f_range[m], jnp.inf)
        
        updates = jnp.full(n, jnp.inf)
        updates = updates.at[1:-1].set(norm_diffs)
        
        is_left_boundary = jnp.concatenate([jnp.array([True]), sorted_r[1:] != sorted_r[:-1]])
        is_right_boundary = jnp.concatenate([sorted_r[:-1] != sorted_r[1:], jnp.array([True])])
        is_boundary = is_left_boundary | is_right_boundary
        
        updates = jnp.where(is_boundary, jnp.inf, updates)
        
        orig_updates = jnp.zeros(n).at[idx].set(updates)
        distances += orig_updates
        
    return distances
