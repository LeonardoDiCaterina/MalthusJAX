"""Tests for TensorNEAT fitness direction, sign conventions, and NaN sentinel handling.

Validates that TensorNeatEvaluator strictly enforces the canonical lower-is-better invariant:
- When maximize=False, raw losses are preserved (lower is better).
- When maximize=True, raw scores are negated (-1.0 * score) so higher raw score yields lower fitness.
- NaN or invalid evaluations are mapped to +inf (worst possible value under minimization).
"""

from typing import Any, Tuple

import jax.numpy as jnp

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.evaluators import TensorNeatEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.genome.real_genome import RealGenome


class DummyTNProblem:
    """Problem returning specified raw fitnesses and empty descriptors."""

    def __init__(self, raw_scores: jnp.ndarray):
        self.raw_scores = raw_scores

    def evaluate(
        self, state: Any, key: Any, forward_fn: Any, pop_member: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        idx = jnp.squeeze(pop_member).astype(jnp.int32)
        score = self.raw_scores[idx]
        return score, jnp.zeros(2)


class DummyTNTransform:
    def transform_population(self, genes: Any, state: Any = None) -> Any:
        # Return indices corresponding to each population member
        return getattr(genes, "values", genes)[0]


def test_tensorneat_evaluator_minimization():
    """When maximize=False, raw problem losses are preserved directly without negation."""
    raw_losses = jnp.array([10.0, 50.0, 5.0, 100.0])
    problem = DummyTNProblem(raw_losses)

    evaluator = TensorNeatEvaluator(
        env=problem,
        transform=DummyTNTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
        forward_fn=lambda x: x,
        maximize=False,
    )

    indices = jnp.arange(4)[:, None]
    pop = BasePopulation(
        genes=RealGenome(values=(indices, indices)),
        fitness=jnp.zeros(4),
    )

    evaluated = evaluator.evaluate_population(pop)

    # In lower-is-better, raw loss 5.0 must be lowest (best)
    assert jnp.allclose(evaluated.fitness, raw_losses)
    best_idx = jnp.argmin(evaluated.fitness)
    assert best_idx == 2  # loss 5.0 is best


def test_tensorneat_evaluator_maximization():
    """When maximize=True, raw problem scores are negated (-score) to minimize."""
    raw_scores = jnp.array([10.0, 50.0, 5.0, 100.0])
    problem = DummyTNProblem(raw_scores)

    evaluator = TensorNeatEvaluator(
        env=problem,
        transform=DummyTNTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
        forward_fn=lambda x: x,
        maximize=True,
    )

    indices = jnp.arange(4)[:, None]
    pop = BasePopulation(
        genes=RealGenome(values=(indices, indices)),
        fitness=jnp.zeros(4),
    )

    evaluated = evaluator.evaluate_population(pop)

    # Raw score 100.0 becomes -100.0, which is lowest (best) under minimization
    assert jnp.allclose(evaluated.fitness, -raw_scores)
    best_idx = jnp.argmin(evaluated.fitness)
    assert best_idx == 3  # score 100.0 is best


def test_tensorneat_evaluator_nan_sentinel_handling():
    """NaN or failed evaluations map to +inf, which is strictly worse than any finite fitness."""
    raw_scores = jnp.array([10.0, jnp.nan, -5.0])
    problem = DummyTNProblem(raw_scores)

    for maximize in (True, False):
        evaluator = TensorNeatEvaluator(
            env=problem,
            transform=DummyTNTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(),
            forward_fn=lambda x: x,
            maximize=maximize,
        )

        indices = jnp.arange(3)[:, None]
        pop = BasePopulation(
            genes=RealGenome(values=(indices, indices)),
            fitness=jnp.zeros(3),
        )

        evaluated = evaluator.evaluate_population(pop)

        # The NaN member must be mapped to +inf
        assert jnp.isposinf(evaluated.fitness[1])
        assert jnp.isfinite(evaluated.fitness[0])
        assert jnp.isfinite(evaluated.fitness[2])
        # Under minimization, +inf is strictly worse than all valid members
        assert evaluated.fitness[0] < evaluated.fitness[1]
        assert evaluated.fitness[2] < evaluated.fitness[1]
