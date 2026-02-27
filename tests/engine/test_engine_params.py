"""Tests for AbstractEngineParams / GeneticEngineParams construction and compute_unroll_num.

Covers:
  - CV-1: __post_init__ removed — unroll_num must stay at user-supplied value.
  - JR-1: compute_unroll_num factory produces correct values.
"""

import pytest

from malthusjax.engine.base import AbstractEngineParams, compute_unroll_num
from malthusjax.engine.genetic_fastengine import GeneticEngineParams


# ---------------------------------------------------------------------------
# compute_unroll_num
# ---------------------------------------------------------------------------


class TestComputeUnrollNum:
    """Unit tests for the compute_unroll_num factory function."""

    def test_standard_100(self):
        assert compute_unroll_num(100) == 10

    def test_small_1(self):
        assert compute_unroll_num(1) == 1

    def test_small_5(self):
        assert compute_unroll_num(5) == 1

    def test_small_10(self):
        assert compute_unroll_num(10) == 1

    def test_large_1000(self):
        assert compute_unroll_num(1000) == 100

    def test_boundary_11(self):
        # 11 // 10 == 1
        assert compute_unroll_num(11) == 1

    def test_boundary_20(self):
        # 20 // 10 == 2
        assert compute_unroll_num(20) == 2

    def test_returns_at_most_num_generations(self):
        # When num_generations is very small, clamp to num_generations itself
        for n in range(1, 5):
            assert compute_unroll_num(n) <= n


# ---------------------------------------------------------------------------
# AbstractEngineParams — no auto-mutation
# ---------------------------------------------------------------------------


class TestAbstractEngineParamsNoAutoMutation:
    """Ensure __post_init__ is gone and unroll_num is never silently overwritten."""

    def test_default_unroll_num_is_1(self):
        params = AbstractEngineParams()
        assert params.unroll_num == 1

    def test_unroll_num_stays_at_user_value(self):
        params = AbstractEngineParams(num_generations=100, unroll_num=5)
        assert params.unroll_num == 5

    def test_unroll_num_stays_at_1_when_not_set(self):
        params = AbstractEngineParams(num_generations=100)
        assert params.unroll_num == 1

    def test_no_post_init_attribute(self):
        """AbstractEngineParams must not define __post_init__."""
        assert not hasattr(AbstractEngineParams, "__post_init__")


# ---------------------------------------------------------------------------
# GeneticEngineParams — same guarantees
# ---------------------------------------------------------------------------


class TestGeneticEngineParamsNoAutoMutation:
    """GeneticEngineParams inherits from AbstractEngineParams and must also be safe."""

    def test_unroll_num_preserved(self):
        params = GeneticEngineParams(num_generations=200, unroll_num=7)
        assert params.unroll_num == 7

    def test_unroll_num_default(self):
        params = GeneticEngineParams(num_generations=200)
        assert params.unroll_num == 1

    def test_no_post_init_attribute(self):
        assert not hasattr(GeneticEngineParams, "__post_init__")
