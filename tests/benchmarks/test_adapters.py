from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from benchmarks.framework.adapters import (
    AbstractBenchmarkAdapter,
    EvosaxAdapter,
    MalthusAdapter,
)


class TestAbstractBenchmarkAdapter:
    """Test the AbstractBenchmarkAdapter interface."""

    def test_abstract_adapter_cannot_be_instantiated(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractBenchmarkAdapter()

    def test_adapter_requires_all_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteAdapter(AbstractBenchmarkAdapter):
            def init(self, rng):
                pass

            # Missing other methods

        with pytest.raises(TypeError):
            IncompleteAdapter()


class TestMalthusAdapter:
    """Test MalthusAdapter implementation."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock GeneticEngine."""
        engine = Mock()

        # Mock init_state
        mock_state = Mock()
        mock_state.best_fitness = 10.0
        engine.init_state = Mock(return_value=mock_state)

        # Mock step - returns (new_state, metrics) tuple
        mock_new_state = Mock()
        mock_new_state.best_fitness = 15.0
        mock_metrics = Mock()
        mock_metrics.best_fitness = 15.0
        engine.step = Mock(return_value=(mock_new_state, mock_metrics))

        return engine

    def test_malthus_adapter_init(self, mock_engine):
        """Test MalthusAdapter initialization."""
        adapter = MalthusAdapter(mock_engine)

        rng = jax.random.PRNGKey(42)
        state = adapter.init(rng)

        mock_engine.init_state.assert_called_once_with(rng)
        assert state is not None

    def test_malthus_adapter_make_step_fn(self, mock_engine):
        """Test MalthusAdapter step function creation."""
        adapter = MalthusAdapter(mock_engine)

        step_fn = adapter.make_step_fn()

        assert callable(step_fn)

        # Test step function execution
        mock_state = Mock()
        # step_fn returns (new_state, _) from engine.step
        new_state, _ = step_fn(mock_state, None)

        mock_engine.step.assert_called_once_with(mock_state)

    def test_malthus_adapter_extract_best_fitness(self, mock_engine):
        """Test MalthusAdapter best fitness extraction.
        Note: MalthusAdapter negates the fitness for BBOB comparison since
        MalthusJAX uses maximize=True (negated BBOB) internally.
        """
        adapter = MalthusAdapter(mock_engine)

        mock_state = Mock()
        mock_state.best_fitness = 42.5

        fitness = adapter.extract_best_fitness(mock_state)

        # Adapter returns -best_fitness for fair comparison with Evosax (minimization)
        assert fitness == -42.5
        assert isinstance(fitness, float)

    def test_malthus_adapter_get_device_info(self, mock_engine):
        """Test MalthusAdapter device information retrieval."""
        adapter = MalthusAdapter(mock_engine)

        device_info = adapter.get_device_info()

        assert isinstance(device_info, str)
        assert len(device_info) > 0


class TestEvosaxAdapter:
    """Test EvosaxAdapter implementation."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock Evosax strategy."""
        strategy = Mock()

        # Mock state
        mock_state = Mock()
        mock_state.best_fitness = 20.0

        # Mock init
        strategy.init = Mock(return_value=mock_state)

        # Mock ask/tell
        strategy.ask = Mock(return_value=(jnp.zeros((50, 10)), mock_state))
        strategy.tell = Mock(return_value=(mock_state, None))

        return strategy

    @pytest.fixture
    def mock_problem(self):
        """Create a mock Evosax problem."""
        problem = Mock()
        problem.num_dims = 10

        # Mock init
        problem.init = Mock(return_value=Mock())

        # Mock eval
        mock_fitness = jnp.ones(50) * 5.0
        problem.eval = Mock(return_value=(mock_fitness, Mock(), None))

        return problem

    @pytest.fixture
    def mock_params(self):
        """Create mock ES parameters."""
        return Mock()

    def test_evosax_adapter_init(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter initialization."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem, pop_size=50)

        rng = jax.random.PRNGKey(42)
        carry = adapter.init(rng)

        # Carry should be a tuple of (state, param_state, rng)
        assert isinstance(carry, tuple)
        assert len(carry) == 3

        mock_problem.init.assert_called_once()

    def test_evosax_adapter_make_step_fn(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter step function creation."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem, pop_size=50)

        step_fn = adapter.make_step_fn()

        assert callable(step_fn)

    def test_evosax_adapter_extract_best_fitness(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter best fitness extraction."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem, pop_size=50)

        mock_state = Mock()
        mock_state.best_fitness = 33.3

        carry = (mock_state, Mock(), Mock())
        fitness = adapter.extract_best_fitness(carry)

        assert fitness == 33.3
        assert isinstance(fitness, float)

    def test_evosax_adapter_get_device_info(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter device information retrieval."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem, pop_size=50)

        device_info = adapter.get_device_info()

        assert isinstance(device_info, str)

    def test_evosax_adapter_pop_size_stored(self, mock_strategy, mock_problem, mock_params):
        """Test that pop_size is stored correctly."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem, pop_size=100)

        assert adapter.pop_size == 100


class TestAdapterComparison:
    """Test that both adapters have consistent interfaces."""

    def test_both_adapters_have_same_methods(self):
        """Test that both adapters implement the same interface methods."""
        malthus_methods = {"init", "make_step_fn", "get_device_info", "extract_best_fitness"}
        evosax_methods = {"init", "make_step_fn", "get_device_info", "extract_best_fitness"}

        # Check MalthusAdapter
        malthus_has = {m for m in malthus_methods if hasattr(MalthusAdapter, m)}
        assert malthus_has == malthus_methods

        # Check EvosaxAdapter
        evosax_has = {m for m in evosax_methods if hasattr(EvosaxAdapter, m)}
        assert evosax_has == evosax_methods

    def test_adapter_return_types_consistent(self):
        """Test that adapters return consistent types."""
        # Create mock adapters
        mock_engine = Mock()
        mock_state = Mock()
        mock_state.best_fitness = 10.0
        mock_engine.init_state = Mock(return_value=mock_state)

        malthus = MalthusAdapter(mock_engine)

        mock_strategy = Mock()
        mock_strategy.init = Mock(return_value=Mock())
        mock_problem = Mock()
        mock_problem.num_dims = 10
        mock_problem.init = Mock(return_value=Mock())

        evosax = EvosaxAdapter(mock_strategy, Mock(), mock_problem, pop_size=50)

        # Both should return float from extract_best_fitness
        assert isinstance(malthus.extract_best_fitness(mock_state), float)

        evosax_carry = (Mock(best_fitness=5.0), Mock(), Mock())
        assert isinstance(evosax.extract_best_fitness(evosax_carry), float)
