import jax
import jax.numpy as jnp
import pytest
from plugins.knapsack_env import KnapsackEnv
from malthusjax.testing.compliance import EnvironmentComplianceSuite

class TestKnapsackEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return KnapsackEnv()
