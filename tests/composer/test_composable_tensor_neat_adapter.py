import jax
import jax.numpy as jnp
import pytest
from src.malthusjax.composer.composable_tensor_neat_adapter import ComposableTensorNEATAdapter
from malthusjax.testing.compliance import AdapterComplianceSuite

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
