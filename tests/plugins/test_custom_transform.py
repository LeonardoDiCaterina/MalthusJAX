import jax
import jax.numpy as jnp
import pytest
from plugins.custom_transform import CustomTransform

class TestCustomTransform:
    @pytest.fixture
    def component(self):
        return CustomTransform()
        
    def test_transform(self, component):
        genome = jnp.zeros(10)
        result = component.transform(genome)
        assert result is not None
