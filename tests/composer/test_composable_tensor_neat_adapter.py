import sys
from unittest.mock import MagicMock
import jax.numpy as jnp
import pytest
import types

from malthusjax.testing.compliance import AdapterComplianceSuite
from malthusjax.composer.composable_tensor_neat_adapter import (
    ComposableTensorNEATAdapter,
    list_algorithms,
    list_genomes,
    list_problems,
    build_composable_tensorneat_engine,
    _tensorneat_native_eval,
)
from malthusjax.composer.adapters import EvalMode


class TestComposableTensorNEATAdapter(AdapterComplianceSuite):
    @pytest.fixture
    def component(self):
        return ComposableTensorNEATAdapter(
            strategy=None,
            params=None,
            pop_size=10,
            num_generations=5,
        )


@pytest.fixture
def mock_tensorneat():
    tn = types.ModuleType("tensorneat")
    tn.algorithm = types.ModuleType("tensorneat.algorithm")
    tn.genome = types.ModuleType("tensorneat.genome")
    tn.problem = types.ModuleType("tensorneat.problem")
    tn.common = types.ModuleType("tensorneat.common")
    
    class SomeAlgo:
        def ask(self): pass
        def tell(self): pass
    
    class SomeGenome:
        def initialize(self): pass
        
    class SomeProblem:
        def evaluate(self): pass
        
    class State:
        def register(self, **kwargs):
            return self
        def update(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self
            
    tn.algorithm.SomeAlgo = SomeAlgo
    tn.genome.SomeGenome = SomeGenome
    tn.problem.SomeProblem = SomeProblem
    tn.common.State = State
    
    sys.modules["tensorneat"] = tn
    sys.modules["tensorneat.algorithm"] = tn.algorithm
    sys.modules["tensorneat.genome"] = tn.genome
    sys.modules["tensorneat.problem"] = tn.problem
    sys.modules["tensorneat.common"] = tn.common
    
    yield tn
    
    if "tensorneat" in sys.modules:
        del sys.modules["tensorneat"]
        del sys.modules["tensorneat.algorithm"]
        del sys.modules["tensorneat.genome"]
        del sys.modules["tensorneat.problem"]
        del sys.modules["tensorneat.common"]


def test_list_functions(mock_tensorneat):
    assert "SomeAlgo" in list_algorithms()
    assert "SomeGenome" in list_genomes()
    assert "SomeProblem" in list_problems()


def test_build_engine_errors():
    with pytest.raises(ValueError, match="is not supported"):
        build_composable_tensorneat_engine(
            algorithm=None, evaluator=None, generations=10, eval_mode=EvalMode.MALTHUSJAX
        )


def test_build_engine_native(mock_tensorneat):
    class DummyAlgo:
        pop_size = 15
    algo = DummyAlgo()
    engine = build_composable_tensorneat_engine(
        algorithm=algo, evaluator=(MagicMock(), MagicMock()), generations=10
    )
    assert engine.pop_size == 15


def test_adapter_init(mock_tensorneat):
    adapter = ComposableTensorNEATAdapter(strategy=None, params=None, pop_size=10, num_generations=5)
    algo = MagicMock()
    algo.setup.return_value = mock_tensorneat.common.State()
    
    state = adapter._adapter_init(
        algorithm=algo, 
        key=jnp.array([0, 0]), 
        params={}, 
        pop_init=(jnp.array([1]), jnp.array([2]))
    )
    assert hasattr(state, "pop_nodes")
    assert hasattr(state, "pop_conns")


def test_adapter_step(mock_tensorneat):
    adapter = ComposableTensorNEATAdapter(strategy=None, params=None, pop_size=10, num_generations=5)
    adapter.maximize = True
    
    algo = MagicMock()
    algo.ask.return_value = jnp.zeros((10, 2))
    algo.transform = lambda s, p: p
    algo.tell.return_value = "new_state"

    class DummyState:
        def update(self, **kwargs):
            return self

    state = DummyState()
    
    def dummy_eval(*args):
        return jnp.ones(10)
    
    new_state, metrics = adapter._adapter_step(
        algorithm=algo, state=state, key=jnp.array([0, 0]), params={},
        evaluator=None, eval_translator=dummy_eval
    )
    assert new_state == "new_state"
    assert "best_fitness_in_generation" in metrics


def test_native_eval(mock_tensorneat):
    problem = MagicMock()
    problem.evaluate = lambda s, k, f, p: jnp.zeros(10)
    
    fitness = _tensorneat_native_eval(
        evaluator=(problem, None),
        state=None,
        transformed_pop=jnp.zeros((10, 2)),
        algorithm=MagicMock(),
        key=jnp.array([0, 0])
    )
    assert fitness.shape == (10,)
