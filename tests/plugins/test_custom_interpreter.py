import jax
import jax.numpy as jnp
import pytest
from plugins.custom_interpreter import CustomInterpreter
from malthusjax.testing.compliance import InterpreterComplianceSuite

class TestCustomInterpreter(InterpreterComplianceSuite):
    @pytest.fixture
    def component(self):
        return CustomInterpreter()

    @pytest.fixture
    def mock_genome(self):
        from malthusjax.core.genome.real_genome import RealGenome
        return RealGenome(values=jnp.ones(10))
        
    @pytest.fixture
    def mock_inputs(self):
        return jnp.zeros(5)
