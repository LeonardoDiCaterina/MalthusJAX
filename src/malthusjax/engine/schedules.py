"""
JAX-native mutation strength schedules.

Provides pure-JAX schedule computation that is safe inside jax.lax.scan.
Replaces Python callable schedules (CV-3/JR-4) which caused
ConcretizationTypeError or silently froze schedule values.

Usage:
    from malthusjax.engine.schedules import ScheduleType, compute_scheduled_strength

    # Inside engine params:
    params = GeneticEngineParams(
        schedule_type=ScheduleType.LINEAR_DECAY,
        initial_strength=1.0,
        final_strength=0.01,
    )

    # Inside jax.lax.scan (safe — all ops are jnp):
    strength = compute_scheduled_strength(
        ScheduleType.LINEAR_DECAY, generation, max_generations,
        initial_strength=1.0, final_strength=0.01,
    )
"""

from __future__ import annotations

import enum

import jax.numpy as jnp


class ScheduleType(enum.IntEnum):
    """Mutation strength schedule type.

    Each variant maps to a row in the vectorized schedule lookup table
    inside ``compute_scheduled_strength``.  Using IntEnum ensures the
    value is a concrete Python int usable as a ``pytree_node=False``
    field — the JIT static fingerprint changes only if the schedule
    *type* changes, not its parameters.
    """

    CONSTANT = 0
    LINEAR_DECAY = 1
    COSINE_ANNEAL = 2
    EXPONENTIAL_DECAY = 3


def compute_scheduled_strength(
    schedule: ScheduleType,
    generation: int,
    max_generations: int,
    initial_strength: float,
    final_strength: float = 0.0,
) -> jnp.ndarray:
    """Compute mutation strength for a given generation.

    This function uses only ``jnp`` operations so that ``generation``
    can be a JAX tracer (e.g. the loop counter inside ``jax.lax.scan``).
    The ``schedule`` and ``max_generations`` arguments are expected to be
    concrete Python values (``pytree_node=False`` fields) and are used
    for the schedule-type dispatch and normalisation respectively.

    Parameters
    ----------
    schedule : ScheduleType
        Which schedule curve to apply.
    generation : int
        Current generation counter (may be a JAX tracer).
    max_generations : int
        Total number of generations (static, concrete int).
    initial_strength : float
        Strength value at generation 0.
    final_strength : float, optional
        Target strength at the final generation (default ``0.0``).
        Ignored for ``CONSTANT`` and ``EXPONENTIAL_DECAY``.

    Returns
    -------
    jnp.ndarray
        Scalar strength value for the current generation.
    """
    t = generation / max_generations
    strength_map = jnp.array(
        [
            initial_strength,  # CONSTANT
            initial_strength + (final_strength - initial_strength) * t,  # LINEAR_DECAY
            final_strength
            + (initial_strength - final_strength) * 0.5 * (1.0 + jnp.cos(jnp.pi * t)),  # COSINE
            initial_strength * jnp.exp(-3.0 * t),  # EXPONENTIAL_DECAY
        ]
    )
    return strength_map[schedule]
