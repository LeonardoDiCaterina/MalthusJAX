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


class TrackBest(enum.IntEnum):
    """Controls how the Hall-of-Fame (best individual) is tracked during
    the ``jax.lax.scan`` evolution loop.

    Selecting a lighter mode reduces scan-carry size and eliminates
    per-step ``jnp.argmax`` / ``Gather`` / ``jax.lax.cond`` overhead.

    ``NONE``
        Zero extra ops per step.  ``best_fitness`` / ``best_genome`` in
        the scan carry are passed through unchanged.  The history's
        ``best_fitness`` reports *per-generation* best (NOT monotonic).
        After ``run()`` completes, a one-shot post-scan finalization
        populates ``best_genome`` and ``best_fitness`` on the final
        state from the last population.  Users can recover the monotonic
        convergence curve via ``jnp.maximum.accumulate(history.best_fitness)``.

    ``LIGHT`` *(default)*
        Per step: one ``jnp.max`` (O(N) reduction) + one ``jnp.maximum``
        (scalar).  Both are fusible with the evaluation kernel.
        ``best_genome`` is NOT tracked in the carry — it is populated
        once after the scan from the final population via ``jnp.argmax``.
        History ``best_fitness`` is monotonically non-decreasing.

    ``FULL``
        Per step: ``jnp.max`` + ``jnp.argmax`` + Gather + element-wise
        ``jnp.where`` on genome leaves.  ``best_genome`` is maintained
        in the carry across all generations.  Uses ``jnp.where`` instead
        of ``jax.lax.cond`` to avoid the XLA fusion barrier.
    """

    NONE = 0
    LIGHT = 1
    FULL = 2


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
    # Python-level branching on schedule type (static dispatch).
    # Since `schedule` is a concrete Python int (IntEnum with pytree_node=False),
    # only the selected branch is traced by JAX, avoiding computation of all 4
    # schedule formulas in the XLA kernel.
    if schedule == ScheduleType.CONSTANT:
        return jnp.asarray(initial_strength)
    
    t = generation / max_generations
    
    if schedule == ScheduleType.LINEAR_DECAY:
        return initial_strength + (final_strength - initial_strength) * t
    elif schedule == ScheduleType.COSINE_ANNEAL:
        return final_strength + (initial_strength - final_strength) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    elif schedule == ScheduleType.EXPONENTIAL_DECAY:
        return initial_strength * jnp.exp(-3.0 * t)
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")
