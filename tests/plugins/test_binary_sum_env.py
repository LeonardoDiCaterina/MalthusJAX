import jax
import jax.numpy as jnp
import pytest
from plugins.binary_sum_env import BinarySumEnv
from malthusjax.testing.compliance import EnvironmentComplianceSuite

class TestBinarySumEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return BinarySumEnv()
