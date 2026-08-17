"""Compatibility layer for evosax 0.1.6.

This module provides mutation and crossover functions that mimic the API of
evosax's algorithms.population_based.simple_ga module, which is only available
in the GitHub development version and not in PyPI releases.

The implementations are extracted from:
https://github.com/RobertTLange/evosax/blob/main/evosax/algorithms/population_based/simple_ga.py
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp


def mutation(key: chex.Array, solution: chex.Array, std: float | chex.Array) -> chex.Array:
    """Gaussian mutation operator.

    Adds Gaussian noise scaled by std to the solution vector.

    Args:
        key: JAX random key
        solution: Current solution vector
        std: Standard deviation (mutation strength)

    Returns:
        Mutated solution
    """
    return solution + std * jax.random.normal(key, solution.shape)


def crossover(
    key: chex.Array,
    parent_1: chex.Array,
    parent_2: chex.Array,
    rate: float,
) -> chex.Array:
    """Uniform crossover operator.

    Each gene is inherited from parent_2 with probability `rate`,
    and from parent_1 with probability (1 - rate).

    Args:
        key: JAX random key
        parent_1: First parent solution
        parent_2: Second parent solution
        rate: Crossover rate (probability of selecting from parent_2)

    Returns:
        Offspring solution
    """
    mask = jax.random.uniform(key, parent_1.shape) < rate
    return jnp.where(mask, parent_2, parent_1)
