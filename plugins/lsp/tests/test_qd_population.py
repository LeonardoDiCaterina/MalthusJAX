import jax
import jax.numpy as jnp
import pytest
from plugins.lsp.populations.qd_population import QDPopulation
from malthusjax.testing.compliance import PopulationComplianceSuite

class TestQDPopulation(PopulationComplianceSuite):
    @pytest.fixture
    def component(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return QDPopulation(genes=genes, fitness=fitness)
