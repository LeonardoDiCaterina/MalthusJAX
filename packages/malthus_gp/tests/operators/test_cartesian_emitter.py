import jax
import jax.numpy as jnp
import pytest
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_cvt_centroids

from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.cartesian_emitter import CartesianEmitter
from malthusjax.testing.compliance import EmitterComplianceSuite


class TestCartesianEmitter(EmitterComplianceSuite):
    @pytest.fixture
    def config(self):
        return CartesianGenomeConfig(num_inputs=1, num_outputs=1, num_rows=1, num_cols=10)

    @pytest.fixture
    def component(self, config):
        return CartesianEmitter(genome_config=config, _batch_size=4)

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
