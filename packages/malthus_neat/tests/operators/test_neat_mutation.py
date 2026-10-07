"""Tests for NeatMutation compliance and execution."""

import jax
import pytest

from malthus_neat.genomes.neat_genome import NeatGenomeConfig, NeatPopulation
from malthus_neat.operators.neat_mutation import NeatMutation
from malthusjax.testing.compliance import MutationComplianceSuite


class TestNeatMutation(MutationComplianceSuite):
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
        return NeatMutation(genome_config=config)

    @pytest.fixture
    def mock_population(self, config):
        key = jax.random.PRNGKey(42)
        return NeatPopulation.init(key, config, size=6)
