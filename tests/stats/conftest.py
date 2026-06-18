"""Shared fixtures for stats layer tests.

Design principle: fixtures produce the EXACT data types that stats functions
consume (MetricVector, PairedSample, RegressionDataset). No DataFrames.
"""
import numpy as np
import pytest

from malthusjax.stats.core import MetricVector, PairedSample


# ─── Factory helpers ────────────────────────────────────────────────────────

def make_mv(name: str = "metric", values: list[float] | None = None, n: int = 30) -> MetricVector:
    """Build a MetricVector from a list or random values."""
    if values is not None:
        return MetricVector(name=name, values=np.array(values, dtype=float))
    rng = np.random.default_rng(42)
    return MetricVector(name=name, values=rng.normal(0, 1, size=n))


def make_pair(
    left: list[float] | None = None,
    right: list[float] | None = None,
    n: int = 30,
    shift: float = 0.0,
    label: str = "test_pair",
) -> PairedSample:
    """Build a PairedSample. If left/right not given, generate with optional shift."""
    rng = np.random.default_rng(42)
    if left is None:
        left_arr = rng.normal(0, 1, size=n)
    else:
        left_arr = np.array(left, dtype=float)
    if right is None:
        right_arr = left_arr + shift + rng.normal(0, 0.1, size=len(left_arr))
    else:
        right_arr = np.array(right, dtype=float)
    return PairedSample(
        left=MetricVector(name="left", values=left_arr),
        right=MetricVector(name="right", values=right_arr),
        label=label,
    )


# ─── Standard fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def paired_equal() -> PairedSample:
    """Paired sample with NO real difference (null hypothesis true)."""
    return make_pair(n=30, shift=0.0)


@pytest.fixture
def paired_shifted() -> PairedSample:
    """Paired sample with a clear location shift."""
    return make_pair(n=30, shift=2.0)


@pytest.fixture
def paired_identical() -> PairedSample:
    """Paired sample where left == right exactly (all diffs = 0)."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    return make_pair(left=vals, right=vals)


@pytest.fixture
def paired_tiny() -> PairedSample:
    """Paired sample with n=2 (minimum for any paired test)."""
    return make_pair(left=[1.0, 2.0], right=[1.5, 2.5])


@pytest.fixture
def paired_single() -> PairedSample:
    """Paired sample with n=1 (below minimum for most tests)."""
    return make_pair(left=[1.0], right=[2.0])

@pytest.fixture
def paired_empty() -> PairedSample:
    """Paired sample with n=0."""
    return make_pair(left=[], right=[])
