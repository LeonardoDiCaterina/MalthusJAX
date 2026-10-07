"""Pytest fixtures for malthus_neat test suite."""

import pytest

from malthus_neat.genomes.neat_genome import NeatGenomeConfig


@pytest.fixture
def base_neat_config():
    return NeatGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
    )
