# mypy: ignore-errors
import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import BaseGenome, BasePopulation


class GenomeComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Genomes.

    To use this suite, inherit from this class and implement the following fixture:
    - `component`: Returns an instantiated genome.
    - `config`: Returns a mock configuration object (or None) required for the genome.
    """

    @pytest.fixture
    def component(self) -> BaseGenome:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def config(self) -> Any:
        return None

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Genome must be a dataclass."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseGenome), "Genome must inherit from BaseGenome."

    def test_size_property(self, component) -> None:
        assert isinstance(component.size, int), "Genome 'size' property must return an int."

    def test_shape_property(self, component) -> None:
        assert isinstance(component.shape, tuple), "Genome 'shape' property must return a tuple."

    def test_autocorrect_jit(self, component, config) -> None:
        """Verify autocorrect can be JIT compiled."""
        jitted_autocorrect = jax.jit(component.autocorrect)
        try:
            corrected = jitted_autocorrect(config)
            assert corrected is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"autocorrect JIT compilation failed due to a tracer leak.\nDetails: {e}")

    def test_distance_jit(self, component) -> None:
        """Verify distance can be JIT compiled and returns a numeric value."""
        # Using component against itself
        jitted_dist = jax.jit(component.distance, static_argnames=("metric",))
        try:
            dist = jitted_dist(component, metric="euclidean")
            # Distance should be a scalar array
            assert dist.ndim == 0
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"distance JIT compilation failed due to a tracer leak.\nDetails: {e}")


class PopulationComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Populations.

    To use this suite, inherit from this class and implement the following fixture:
    - `component`: Returns an instantiated population (e.g., size 10).
    """

    @pytest.fixture
    def component(self) -> BasePopulation:
        raise NotImplementedError("You must implement the `component` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Population must be a dataclass."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BasePopulation), "Population must inherit from BasePopulation."

    def test_length(self, component) -> None:
        """Verify __len__ works and matches fitness shape."""
        assert len(component) > 0, "Mock population should have a size > 0."
        assert len(component) == component.fitness.shape[0]

    def test_slicing(self, component) -> None:
        """Verify __getitem__ works for integer and slice indexing."""
        pop_size = len(component)
        
        # Slice indexing
        sliced_pop = component[: pop_size // 2]
        assert isinstance(sliced_pop, BasePopulation)
        assert len(sliced_pop) == pop_size // 2
        
        # Single element indexing
        single_ind = component[0]
        # Should return a single genome instance, not a population
        assert not isinstance(single_ind, BasePopulation)

    def test_clone_buffers(self, component) -> None:
        """Verify clone_buffers deep-copies arrays."""
        cloned = component.clone_buffers()
        assert cloned is not component
        # Modify original fitness, ensure cloned fitness doesn't change
        # (This is just a basic structure check)
        assert cloned.fitness.shape == component.fitness.shape

    def test_spawn_offspring(self, component) -> None:
        """Verify spawn_offspring resets info dict and creates NaN fitness by default."""
        offspring_genes = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), component.genes)
        
        # Add mock info
        component = dataclasses.replace(component, info={"test_key": jnp.zeros(len(component))})
        
        offspring_pop = component.spawn_offspring(offspring_genes)
        assert isinstance(offspring_pop, BasePopulation)
        assert not offspring_pop.info  # info should be reset to empty dict by default
        
        # Check that default fitness is NaN
        assert jnp.isnan(offspring_pop.fitness).all()
