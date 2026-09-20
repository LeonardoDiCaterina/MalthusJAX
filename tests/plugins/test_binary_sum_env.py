import pytest

from malthusjax.testing.compliance import EnvironmentComplianceSuite
from plugins.binary_sum_env import BinarySumEnv


class TestBinarySumEnv(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return BinarySumEnv()
