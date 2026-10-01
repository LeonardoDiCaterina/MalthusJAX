"""Tests for NeatCrossover compliance and execution."""

import jax
import pytest

from malthus_neat.genomes.neat_genome import NeatGenomeConfig, NeatPopulation
from malthus_neat.operators.neat_crossover import NeatCrossover
from malthusjax.testing.compliance import CrossoverComplianceSuite


class TestNeatCrossover(CrossoverComplianceSuite):
    @pytest.fixture
    def config(self):
        return NeatGenomeConfig(
            num_inputs=2,
            num_outputs=1,
            max_nodes=10,
            max_conns=20,
        )

    @pytest.fixture
    def component(self, config):
        return NeatCrossover(genome_config=config)

    @pytest.fixture
    def mock_pop1(self, config):
        key = jax.random.PRNGKey(10)
        return NeatPopulation.init(key, config, size=6)

    @pytest.fixture
    def mock_pop2(self, config):
        key = jax.random.PRNGKey(20)
        return NeatPopulation.init(key, config, size=6)
