import jax
import jax.numpy as jnp
import pytest
from plugins.tensor_neat_transform import TensorNeatTransform

class TestTensorNeatTransform:
    @pytest.fixture
    def component(self):
        return TensorNeatTransform()
        
    def test_transform(self, component):
        genome = jnp.zeros(10)
        result = component.transform(genome)
        assert result is not None
