"""Tests for NeatGenome and NeatPopulation compliance."""

import jax
import pytest

from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig, NeatPopulation
from malthusjax.testing.compliance import GenomeComplianceSuite


class TestNeatGenome(GenomeComplianceSuite):
    @pytest.fixture
    def component(self):
        config = NeatGenomeConfig(
            num_inputs=2,
            num_outputs=1,
            max_nodes=10,
            max_conns=20,
        )
        key = jax.random.PRNGKey(42)
        return NeatGenome.random_init(key, config)


def test_neat_active_masks_and_density():
    config = NeatGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
    )
    key = jax.random.PRNGKey(101)
    genome = NeatGenome.random_init(key, config)

    active_nodes, active_conns = genome.get_active_mask()
    assert active_nodes.shape == (10,)
    assert active_conns.shape == (20,)

    n_nodes = genome.num_active_nodes
    n_conns = genome.num_active_conns
    density = genome.connection_density

    assert n_nodes >= 3  # At least 2 inputs + 1 output
    assert n_conns >= 1
    assert 0.0 < float(density) <= 1.0


def test_neat_population_init():
    config = NeatGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
    )
    key = jax.random.PRNGKey(7)
    pop = NeatPopulation.init(key, config, size=8)

    assert pop.genes.nodes.shape == (8, 10, 5)
    assert pop.genes.conns.shape == (8, 20, 3)
    assert pop.fitness.shape == (8,)
    assert len(pop) == 8
