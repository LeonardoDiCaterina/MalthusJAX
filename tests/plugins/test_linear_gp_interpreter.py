import jax
import jax.numpy as jnp
import pytest
from plugins.linear_gp_interpreter import LinearGPInterpreter
from malthusjax.testing.compliance import InterpreterComplianceSuite

class TestLinearGPInterpreter(InterpreterComplianceSuite):
    @pytest.fixture
    def component(self):
        return LinearGPInterpreter()

    @pytest.fixture
    def mock_genome(self):
        from malthusjax.core.genome.real_genome import RealGenome
        return RealGenome(values=jnp.ones(10))
        
    @pytest.fixture
    def mock_inputs(self):
        return jnp.zeros(5)
