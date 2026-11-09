"""Archive reductions in higher-is-better repertoire space, without an offset."""

import jax.numpy as jnp


def archive_metrics(fitnesses):
    """Empty cells are -inf. Empty objective statistics are NaN, coverage is zero.

    NaN/+inf are invalid archive data, not additional empty cells. Their presence
    makes all reported statistics unavailable. Dispersion uses population ddof=0.
    """
    occupied = jnp.isfinite(fitnesses)
    invalid = jnp.any(~occupied & (fitnesses != -jnp.inf))
    count = jnp.sum(occupied)
    total = jnp.sum(jnp.where(occupied, fitnesses, 0.0))
    mean = total / jnp.maximum(count, 1)
    variance = jnp.sum(jnp.where(occupied, (fitnesses - mean) ** 2, 0.0)) / jnp.maximum(count, 1)
    available = (count > 0) & ~invalid
    def value(x):
        return jnp.where(available, x, jnp.nan)
    return {
        "max_fitness": value(jnp.max(fitnesses)),
        "mean_fitness": value(mean),
        "std_fitness": value(jnp.sqrt(variance)),
        "qd_score": value(total),
        "coverage": jnp.where(invalid, jnp.nan, 100.0 * count / fitnesses.size),
        "occupied_cells": count,
    }
