import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.fitness.real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)
from malthusjax.core.genome.real_genome import RealGenome


@pytest.fixture
def key():
    return jr.PRNGKey(42)

@pytest.fixture
def genome_zero():
    """A genome at the origin."""
    return RealGenome(values=jnp.zeros(5))

@pytest.fixture
def genome_ones():
    """A genome of all ones."""
    return RealGenome(values=jnp.ones(5))

def test_sphere_evaluator(genome_zero, genome_ones):
    """Test Sphere function (f(x) = sum(x^2))."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config)

    # Minima at 0
    score_zero = evaluator.evaluate(genome_zero)
    assert score_zero == 0.0 # -0.0

    # At 1s: sum(1^2 * 5) = 5. Since maximize=False, return -5
    score_ones = evaluator.evaluate(genome_ones)
    assert score_ones == -5.0

    # Test Maximization mode (should return positive)
    config_max = SphereConfig(maximize=True)
    evaluator_max = SphereEvaluator(config=config_max)
    assert evaluator_max.evaluate(genome_ones) == 5.0

def test_griewank_evaluator(genome_zero):
    """Test Griewank function."""
    config = GriewankConfig(maximize=False)
    evaluator = GriewankEvaluator(config=config)

    # Global minima at 0, f(0) = 0.
    # Formula: 1 + sum/4000 - prod(cos)
    # At 0: 1 + 0 - 1 = 0
    score = evaluator.evaluate(genome_zero)
    # Floating point precision might be tiny, check close to 0
    assert jnp.abs(score) < 1e-6

    # Test maximization
    config_max = GriewankConfig(maximize=True)
    eval_max = GriewankEvaluator(config=config_max)
    assert eval_max.evaluate(genome_zero) == 0.0

def test_box_evaluator_constraints(genome_zero):
    """Test Box constraints and penalties."""
    # Target is at [2, 2, 2, 2, 2]
    # Box is [1, 3] for all dims
    target = jnp.full(5, 2.0)
    lower = jnp.full(5, 1.0)
    upper = jnp.full(5, 3.0)

    config = BoxConfig(
        maximize=True, # Note: BoxEvaluator usually returns negative distance
        target_point=target,
        box_bounds=(lower, upper),
        objective_type="distance"
    )
    evaluator = BoxEvaluator(config=config)

    # 1. Genome at 0 (Outside bounds [1, 3])
    # Distance to target (2): sqrt(5 * 2^2) = sqrt(20) approx 4.47
    # Violation: lower bound is 1, genome is 0. Violation = 1.0 per dim * 5 = 5.0
    # Penalty: 5.0 * 1000 = 5000
    # Expected: -4.47 - 5000 = -5004.47
    score_bad = evaluator.evaluate(genome_zero)
    assert score_bad < -1000.0 # Huge penalty

    # 2. Genome at Target (Inside bounds)
    # Distance 0, Penalty 0
    genome_target = RealGenome(values=target)
    score_perfect = evaluator.evaluate(genome_target)
    assert score_perfect == 0.0

def test_box_evaluator_factory(key):
    """Test the static factory method."""
    config = BoxEvaluator.create_random_problem(key, dimensions=10)
    assert config.target_point.shape == (10,)
    assert isinstance(config, BoxConfig)

def test_box_objective_types(genome_ones):
    """Test alternate objective types."""
    target = jnp.zeros(5)
    lower = jnp.full(5, -5.0)
    upper = jnp.full(5, 5.0)

    # Sphere objective
    config = BoxConfig(
        maximize=True, target_point=target, box_bounds=(lower, upper),
        objective_type="sphere"
    )
    evaluator = BoxEvaluator(config=config)

    # Genome at 1, Target 0. Sphere dist = 5. Negated = -5.
    score = evaluator.evaluate(genome_ones)
    assert score == -5.0
