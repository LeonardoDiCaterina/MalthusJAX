import pytest

from malthusjax.testing.compliance import EnvironmentComplianceSuite
from plugins.custom_rl_env import CustomRLEnv


class TestCustomRLEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return CustomRLEnv()
