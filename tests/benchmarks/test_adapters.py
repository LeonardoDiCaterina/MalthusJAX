import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock, MagicMock

from benchmarks.framework.adapters import (
    AbstractBenchmarkAdapter,
    MalthusAdapter,
    EvosaxAdapter,
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
        
        # Mock step
        mock_metrics = Mock()
        mock_metrics.best_fitness = 15.0
        engine.step = Mock(return_value=(mock_state, mock_metrics))
        
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
        new_state, metric = step_fn(mock_state, None)
        
        mock_engine.step.assert_called_once_with(mock_state)
        assert metric == 15.0  # From mock_metrics.best_fitness
    
    def test_malthus_adapter_get_best_fitness(self, mock_engine):
        """Test MalthusAdapter best fitness extraction."""
        adapter = MalthusAdapter(mock_engine)
        
        mock_state = Mock()
        mock_state.best_fitness = 42.5
        
        fitness = adapter.get_best_fitness(mock_state)
        
        assert fitness == 42.5
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
        strategy.population_size = 50
        
        # Mock state
        mock_state = Mock()
        mock_state.best_fitness = 20.0
        
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
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem)
        
        rng = jax.random.PRNGKey(42)
        carry = adapter.init(rng)
        
        # Carry should be a tuple of (state, param_state, rng)
        assert isinstance(carry, tuple)
        assert len(carry) == 3
        
        mock_problem.init.assert_called_once()
    
    def test_evosax_adapter_make_step_fn(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter step function creation."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem)
        
        step_fn = adapter.make_step_fn()
        
        assert callable(step_fn)
    
    def test_evosax_adapter_get_best_fitness(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter best fitness extraction."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem)
        
        mock_state = Mock()
        mock_state.best_fitness = 33.3
        
        carry = (mock_state, Mock(), Mock())
        fitness = adapter.get_best_fitness(carry)
        
        assert fitness == 33.3
        assert isinstance(fitness, float)
    
    def test_evosax_adapter_get_device_info(self, mock_strategy, mock_problem, mock_params):
        """Test EvosaxAdapter device information retrieval."""
        adapter = EvosaxAdapter(mock_strategy, mock_params, mock_problem)
        
        device_info = adapter.get_device_info()
        
        assert isinstance(device_info, str)
    
    def test_evosax_adapter_missing_population_size(self, mock_problem, mock_params):
        """Test error handling when strategy lacks population_size."""
        strategy = Mock()
        # Don't set population_size or pop_size
        del strategy.population_size
        del strategy.pop_size
        
        adapter = EvosaxAdapter(strategy, mock_params, mock_problem)
        
        with pytest.raises(AttributeError):
            rng = jax.random.PRNGKey(42)
            adapter.init(rng)
    
    def test_evosax_adapter_missing_num_dims(self, mock_strategy, mock_params):
        """Test error handling when problem lacks num_dims."""
        problem = Mock()
        # Don't set num_dims
        del problem.num_dims
        
        adapter = EvosaxAdapter(mock_strategy, mock_params, problem)
        
        with pytest.raises(AttributeError):
            rng = jax.random.PRNGKey(42)
            adapter.init(rng)


class TestAdapterComparison:
    """Test that both adapters follow the same interface."""
    
    def test_both_adapters_have_same_methods(self):
        """Test that MalthusAdapter and EvosaxAdapter have the same public methods."""
        malthus_methods = set(dir(MalthusAdapter))
        evosax_methods = set(dir(EvosaxAdapter))
        
        # Core interface methods
        required_methods = {
            'init',
            'make_step_fn',
            'get_best_fitness',
            'get_device_info',
        }
        
        assert required_methods.issubset(malthus_methods)
        assert required_methods.issubset(evosax_methods)
    
    def test_adapter_return_types_consistent(self):
        """Test that adapters return consistent types."""
        # This is a structural test - both should follow the same patterns
        
        mock_engine = Mock()
        mock_engine.init_state = Mock(return_value=Mock(best_fitness=10.0))
        mock_engine.step = Mock(return_value=(Mock(best_fitness=15.0), Mock(best_fitness=15.0)))
        
        malthus = MalthusAdapter(mock_engine)
        
        # Both should return the same types
        rng = jax.random.PRNGKey(42)
        m_state = malthus.init(rng)
        assert m_state is not None
        
        m_step = malthus.make_step_fn()
        assert callable(m_step)
        
        m_fitness = malthus.get_best_fitness(m_state)
        assert isinstance(m_fitness, float)
        
        m_device = malthus.get_device_info()
        assert isinstance(m_device, str)
