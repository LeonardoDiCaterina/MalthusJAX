import jax.numpy as jnp
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.testing.compliance import MutationComplianceSuite
from plugins.compliance_mutation import ComplianceMutation


class TestComplianceMutation(MutationComplianceSuite):
    @pytest.fixture
    def component(self):
        return ComplianceMutation()

    @pytest.fixture
    def mock_population(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return BasePopulation(genes=genes, fitness=fitness)
