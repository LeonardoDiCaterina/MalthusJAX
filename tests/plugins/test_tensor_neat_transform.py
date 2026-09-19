import jax.numpy as jnp
import pytest

from plugins.tensor_neat_transform import TensorNeatTransform


class TestTensorNeatTransform:
    @pytest.fixture
    def component(self):
        class MockAlgorithm:
            def transform(self, state, genes):
                return genes

        return TensorNeatTransform(algorithm=MockAlgorithm())

    def test_transform(self, component):
        class MockGenome:
            values = (jnp.zeros(10), jnp.zeros(10))

        genome = MockGenome()
        result = component.transform(genome)
        assert result is not None
