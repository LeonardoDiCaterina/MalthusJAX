import jax
import pytest
from plugins.custom_evaluator import create_custom_evaluator

def test_custom_evaluator_creation():
    """Test that the evaluator composition builds successfully."""
    evaluator = create_custom_evaluator()
    assert evaluator is not None
