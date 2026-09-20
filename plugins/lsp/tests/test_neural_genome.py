import jax
import jax.numpy as jnp
import pytest
from plugins.lsp.genomes.neural_genome import NeuralGenome
from malthusjax.testing.compliance import GenomeComplianceSuite

class TestNeuralGenome(GenomeComplianceSuite):
    @pytest.fixture
    def component(self):
        rng = jax.random.PRNGKey(0)
        return NeuralGenome.random_init(rng, config=None)
