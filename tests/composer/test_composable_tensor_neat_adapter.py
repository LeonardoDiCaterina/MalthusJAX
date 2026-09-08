import pytest

from malthusjax.testing.compliance import AdapterComplianceSuite
from malthusjax.composer.composable_tensor_neat_adapter import ComposableTensorNEATAdapter


class TestComposableTensorNEATAdapter(AdapterComplianceSuite):
    @pytest.fixture
    def component(self):
        # The adapter decorator changes the class signature.
        # We need to instantiate it with dummy args expected by the universal engine base.
        return ComposableTensorNEATAdapter(
            strategy=None,
            params=None,
            pop_size=10,
            num_generations=5,
        )
