"""Tests for shared initial population generation utility."""

import jax.numpy as jnp
import numpy as np

from malthusjax.composer.backends._population_init import generate_initial_population
from malthusjax.composer.experiment_config import (
    ExperimentConfig,
    MalthusJAXBackendConfig,
    PopulationConfig,
)


def test_uniform_population_shape_and_bounds():
    """Uniform random population has expected shape and stays within bounds."""
    config = ExperimentConfig(
        population=PopulationConfig(size=30, genome_length=8, bounds=(-2.0, 3.0))
    )
    pop = generate_initial_population(config, pop_seed=42)
    assert pop is not None
    assert pop.shape == (30, 8)
    assert float(jnp.min(pop)) >= -2.0
    assert float(jnp.max(pop)) <= 3.0


def test_deterministic_across_calls():
    """Identical seeds produce identical initial populations."""
    config = ExperimentConfig(
        population=PopulationConfig(size=20, genome_length=5, bounds=(-5.0, 5.0))
    )
    pop1 = generate_initial_population(config, pop_seed=123)
    pop2 = generate_initial_population(config, pop_seed=123)
    assert pop1 is not None and pop2 is not None
    np.testing.assert_array_equal(np.asarray(pop1), np.asarray(pop2))


def test_different_seeds_produce_different_populations():
    """Different seeds produce different initial populations."""
    config = ExperimentConfig(
        population=PopulationConfig(size=20, genome_length=5, bounds=(-5.0, 5.0))
    )
    pop1 = generate_initial_population(config, pop_seed=1)
    pop2 = generate_initial_population(config, pop_seed=2)
    assert not np.array_equal(np.asarray(pop1), np.asarray(pop2))


def test_dict_config_backward_compatibility():
    """Legacy dictionary configs are supported correctly."""
    legacy_cfg = {
        "pop_size": 25,
        "genome_length": 6,
        "bounds": (-1.0, 1.0),
    }
    pop = generate_initial_population(legacy_cfg, pop_seed=99)
    assert pop is not None
    assert pop.shape == (25, 6)
    assert float(jnp.min(pop)) >= -1.0
    assert float(jnp.max(pop)) <= 1.0


def test_bbob_population_sampling():
    """BBOB fitness spec uses BBOB problem sampling."""
    config = ExperimentConfig(
        population=PopulationConfig(size=10, genome_length=5),
        backend=MalthusJAXBackendConfig(fitness="bbob:fn_name=sphere,dim=5"),
    )
    pop = generate_initial_population(config, pop_seed=42)
    assert pop is not None
    assert pop.shape == (10, 5)
