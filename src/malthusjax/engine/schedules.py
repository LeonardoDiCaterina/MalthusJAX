"""
JAX-native mutation strength schedules.

Provides pure-JAX schedule computation that is safe inside jax.lax.scan.
Replaces Python callable schedules (CV-3/JR-4) which caused
ConcretizationTypeError or silently froze schedule values.

Usage::

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


class TrackMetrics(enum.IntEnum):
    """Controls how fitness metrics are tracked during the evolution loop.

    ``NONE``
        No metrics are tracked. mean_fitness and std_fitness are dummy zeros.
        Allows XLA's Dead Code Elimination (DCE) to prune mean/std reduction operations.

    ``BASIC``
        Tracks mean_fitness, but std_fitness is dummy zero.

    ``ALL``
        Tracks both mean_fitness and std_fitness.
    """

    NONE = 0
    BASIC = 1
    ALL = 2


def compute_scheduled_strength(
    schedule: ScheduleType,
    generation: int,
    max_generations: int,
    initial_strength: float,
    final_strength: float = 0.0,
) -> jnp.ndarray:
    """Return the mutation strength corresponding to a generation.

    The implementation relies exclusively on ``jnp`` operations so that
    ``generation`` may be a tracer inside a JAX scan. The schedule type and
    max generations are expected to be concrete constants used to select and
    normalise the curve.

    Supported schedules include constant, linear decay, cosine annealing and
    exponential decay. ``final_strength`` is ignored for types that do not
    depend on it.
    """
    if schedule == ScheduleType.CONSTANT:
        return jnp.asarray(initial_strength)

    # Normalise generation to [0, 1] range across the entire run.
    # We use max(1, ...) to avoid division by zero if max_generations=1
    t = jnp.asarray(generation) / jnp.maximum(1, max_generations - 1)

    if schedule == ScheduleType.LINEAR_DECAY:
        return jnp.asarray(initial_strength + (final_strength - initial_strength) * t)
    elif schedule == ScheduleType.COSINE_ANNEAL:
        return jnp.asarray(
            final_strength + (initial_strength - final_strength) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        )
    elif schedule == ScheduleType.EXPONENTIAL_DECAY:
        return jnp.asarray(initial_strength * jnp.exp(-3.0 * t))
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")
