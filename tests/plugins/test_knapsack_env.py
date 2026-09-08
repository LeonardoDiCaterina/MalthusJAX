import pytest

from malthusjax.testing.compliance import EnvironmentComplianceSuite
from plugins.knapsack_env import KnapsackEnv


class TestKnapsackEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return KnapsackEnv()
