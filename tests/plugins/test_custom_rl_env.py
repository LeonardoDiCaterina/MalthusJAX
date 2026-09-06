import jax
import jax.numpy as jnp
import pytest
from plugins.custom_rl_env import CustomRLEnv
from malthusjax.testing.compliance import EnvironmentComplianceSuite

class TestCustomRLEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return CustomRLEnv()
