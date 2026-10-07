"""Tests for NeatEmitter compliance and execution."""

import jax
import jax.numpy as jnp
import pytest
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_cvt_centroids

from malthus_neat.genomes.neat_genome import NeatGenomeConfig
from malthus_neat.operators.neat_emitter import NeatEmitter
from malthusjax.testing.compliance import EmitterComplianceSuite


class TestNeatEmitter(EmitterComplianceSuite):
    @pytest.fixture
    def config(self):
        return NeatGenomeConfig(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=16,
        )

    @pytest.fixture
    def component(self, config):
        return NeatEmitter(genome_config=config, _batch_size=4)

    @pytest.fixture
    def mock_repertoire(self, config):
        centroids = compute_cvt_centroids(
            num_descriptors=2,
            num_init_cvt_samples=100,
            num_centroids=5,
            minval=[0.0, 0.0],
            maxval=[1.0, 1.0],
            key=jax.random.PRNGKey(0),
        )
        init_pop = config.init_population(jax.random.PRNGKey(1), 5)
        repertoire = MapElitesRepertoire.init(
            genotypes=init_pop.genes,
            fitnesses=-jnp.ones(5),
            descriptors=jnp.zeros((5, 2)),
            centroids=centroids,
        )
        return repertoire
