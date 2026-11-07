import sys
from typing import Any
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("tensorneat")
import tensorneat as tn
from flax import struct

from malthusjax.composer.tensorneat_adapter import build_tensorneat_engine

from .base_adapter_suite import BaseAdapterTestSuite


@struct.dataclass
class MockState:
    data: dict[str, Any] = struct.field(default_factory=dict)

    def register(self, **kwargs):
        new_data = self.data.copy()
        new_data.update(kwargs)
        return MockState(new_data)

    def update(self, **kwargs):
        return self.register(**kwargs)


@pytest.fixture
def mock_tensorneat_modules(monkeypatch):
    mock_tensorneat = MagicMock()
    mock_state_module = MagicMock()
    mock_state_module.State = MockState
    mock_tensorneat.common = mock_state_module

    monkeypatch.setitem(sys.modules, "tensorneat", mock_tensorneat)
    monkeypatch.setitem(sys.modules, "tensorneat.common", mock_state_module)
    yield


class MockTensorNEATAlgorithm:
    def __init__(self, pop_size=10):
        self.pop_size = pop_size

    def setup(self, state):
        return state.register(generation=0)

    def ask(self, state):
        key = state.data["randkey"]
        return jax.random.uniform(key, (self.pop_size, 5))

    def transform(self, state, pop):
        # mock transform: just multiply by 2
        return pop * 2.0

    def forward(self, state, params, inputs):
        # mock forward: sum(params * inputs)
        return jnp.sum(params * inputs)

    def tell(self, state, fitness):
        gen = state.data["generation"]
        return state.update(generation=gen + 1)


class MockProblem:
    def evaluate(self, state, key, forward, transformed_pop):
        # simple mock evaluation: sum of parameters
        return jnp.sum(transformed_pop)


@pytest.mark.usefixtures("mock_tensorneat_modules")
class TestTensorNEATAdapterMocked(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = True, eval_mode: str = "native", seed: int = 0):
        if eval_mode != "native":
            pytest.skip("TensorNEAT adapter does not support EvalMode.MALTHUSJAX yet.")

        algorithm = MockTensorNEATAlgorithm(pop_size=10)
        problem = MockProblem()

        return build_tensorneat_engine(
            algorithm=algorithm,
            evaluator=problem,
            generations=3,
            pop_size=10,
            maximize=maximize,
            eval_mode=eval_mode,
        )

    def test_maximize_flag_changes_outcome(self):
        pytest.skip(
            "TensorNEAT native mode uses native evaluator which always maximizes. Maximize flag only affects metrics."
        )

    def test_maximize_history_changes(self):
        pytest.skip(
            "Skipping because TensorNEAT native mode mock doesn't support generic maximization."
        )


class TestTensorNEATAdapterRealIntegration(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        if eval_mode != "native":
            return None

        algorithm = tn.algorithm.NEAT(
            pop_size=10,
            species_size=5,
            genome=tn.genome.DefaultGenome(
                num_inputs=2,
                num_outputs=1,
            ),
        )
        problem = tn.problem.XOR()

        return build_tensorneat_engine(
            algorithm=algorithm,
            evaluator=problem,
            generations=3,
            pop_size=10,
            maximize=maximize,
            eval_mode=eval_mode,
        )
