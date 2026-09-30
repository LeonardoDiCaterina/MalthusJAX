"""Tests verifying shared initial population parity and determinism in compare()."""
from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from malthusjax.composer.composer import Composer


def test_shared_population_determinism():
    """Two pipelines receiving the same pop_seed must get bitwise-identical initial populations."""
    composer = Composer()
    pop1 = composer._generate_initial_population(
        {"pop_size": 10, "genome_length": 5, "bounds": (-5.0, 5.0)}, pop_seed=123
    )
    pop2 = composer._generate_initial_population(
        {"pop_size": 10, "genome_length": 5, "bounds": (-5.0, 5.0)}, pop_seed=123
    )
    assert jnp.array_equal(pop1, pop2)


def test_shared_population_bbob_path():
    """BBOB initial population uses the problem sample() method."""
    composer = Composer()
    pop = composer._generate_initial_population(
        {"pop_size": 4, "fitness": "bbob:fn=sphere,dim=3", "bounds": (-5.0, 5.0)},
        pop_seed=42,
    )
    assert pop.shape == (4, 3)


def test_compare_injects_same_population_to_all_pipelines():
    """All pipelines in a compare() call must receive the exact same initial_population."""
    injected_pops = []
    original_quick_run = Composer.quick_run

    def spy_quick_run(self, **kwargs):
        injected_pops.append(kwargs.get("initial_population"))
        return original_quick_run(self, **kwargs)

    with patch.object(Composer, "quick_run", spy_quick_run):
        composer = Composer()
        composer.compare(
            pipelines={"A": {}, "B": {"crossover": "blend:alpha=0.9"}},
            fitness="sphere:dim=2",
            pop_size=4,
            generations=1,
            seeds=(1,),
            shared_initial_population=True,
        )

    assert len(injected_pops) == 2
    assert injected_pops[0] is not None
    assert jnp.array_equal(injected_pops[0], injected_pops[1])
