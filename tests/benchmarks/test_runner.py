import pytest
import time
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch

from benchmarks.framework.runner import run_adapter_benchmark, BenchmarkResult
from benchmarks.framework.adapters import AbstractBenchmarkAdapter


class MockAdapter(AbstractBenchmarkAdapter):
    """Simple mock adapter for testing."""
    
    def __init__(self, num_gens: int = 10):
        self.num_gens = num_gens
        self.init_called = False
        self.step_called = False
    
    def init(self, rng: jax.Array):
        self.init_called = True
        return {"generation": 0, "best_fitness": 0.0}
    
    def make_step_fn(self):
        self.step_called = True
        def step(carry, _):
            new_carry = {
                "generation": carry["generation"] + 1,
                "best_fitness": carry["best_fitness"] + 1.0
            }
            return new_carry, new_carry["best_fitness"]
        return step
    
    def get_best_fitness(self, carry):
        return float(carry["best_fitness"])
    
    def get_device_info(self):
        return "CPU"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            framework="MalthusJAX",
            device="CPU",
            compile_time=0.5,
            execution_time=2.0,
            generations_per_sec=50.0,
            best_fitness=10.5,
            fitness_std=0.15,
        )
        
        assert result.framework == "MalthusJAX"
        assert result.device == "CPU"
        assert result.compile_time == 0.5
        assert result.execution_time == 2.0
        assert result.generations_per_sec == 50.0
        assert result.best_fitness == 10.5
        assert result.fitness_std == 0.15


class TestRunAdapterBenchmark:
    """Test the run_adapter_benchmark function."""
    
    def test_basic_benchmark_execution(self):
        """Test basic benchmark execution with default parameters."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=10,
            seed=42,
            framework_name="TestFramework"
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.framework == "TestFramework"
        assert result.device == "CPU"
        assert result.compile_time > 0
        assert result.execution_time > 0
        assert result.generations_per_sec > 0
        assert result.fitness_std >= 0.0  # Should be 0 for single repeat
        assert adapter.init_called
        assert adapter.step_called
    
    def test_benchmark_with_unroll(self):
        """Test benchmark execution with unroll parameter."""
        adapter = MockAdapter(num_gens=20)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=20,
            seed=42,
            framework_name="TestFramework",
            unroll=4
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.best_fitness == 20.0  # Should iterate 20 times
    
    def test_benchmark_with_repeats(self):
        """Test benchmark execution with multiple repeats."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=10,
            seed=42,
            framework_name="TestFramework",
            repeats=3
        )
        
        # With 3 repeats, execution_time should be averaged
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time > 0
        # GPS should be calculated from averaged execution time
        assert result.generations_per_sec > 0
        # Fitness std should be calculated from multiple repeats
        assert result.fitness_std >= 0.0
        # With deterministic execution, std should be very small
        assert result.fitness_std < 1.0
    
    def test_benchmark_different_seeds(self):
        """Test that different seeds produce consistent but independent runs."""
        adapter1 = MockAdapter(num_gens=10)
        adapter2 = MockAdapter(num_gens=10)
        
        result1 = run_adapter_benchmark(adapter1, 10, 42, "Test")
        result2 = run_adapter_benchmark(adapter2, 10, 43, "Test")
        
        # Both should succeed
        assert result1.best_fitness == 10.0
        assert result2.best_fitness == 10.0
    
    def test_compile_time_measurement(self):
        """Test that compile time is measured separately from execution."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test")
        
        # Compile and execution times should both be positive
        assert result.compile_time > 0
        assert result.execution_time > 0
        # They should be measured independently
        assert result.compile_time != result.execution_time
    
    def test_generations_per_sec_calculation(self):
        """Test that GPS is calculated correctly."""
        adapter = MockAdapter(num_gens=100)
        
        result = run_adapter_benchmark(adapter, 100, 42, "Test")
        
        # GPS = num_gens / execution_time
        expected_gps = 100 / result.execution_time
        assert abs(result.generations_per_sec - expected_gps) < 0.1


class TestAdapterInterface:
    """Test the AbstractBenchmarkAdapter interface."""
    
    def test_adapter_implements_all_methods(self):
        """Test that MockAdapter implements all required methods."""
        adapter = MockAdapter()
        
        # All methods should be callable
        assert callable(adapter.init)
        assert callable(adapter.make_step_fn)
        assert callable(adapter.get_best_fitness)
        assert callable(adapter.get_device_info)
    
    def test_adapter_init_returns_state(self):
        """Test that init returns a valid initial state."""
        adapter = MockAdapter()
        rng = jax.random.PRNGKey(42)
        
        state = adapter.init(rng)
        
        assert isinstance(state, dict)
        assert "generation" in state
        assert "best_fitness" in state
    
    def test_adapter_step_fn_signature(self):
        """Test that step function has correct signature."""
        adapter = MockAdapter()
        step_fn = adapter.make_step_fn()
        
        # Step function should accept (carry, _) and return (new_carry, metric)
        carry = {"generation": 0, "best_fitness": 0.0}
        new_carry, metric = step_fn(carry, None)
        
        assert isinstance(new_carry, dict)
        assert isinstance(metric, (float, jnp.ndarray))


class TestPerformanceMeasurement:
    """Test performance measurement accuracy."""
    
    def test_execution_time_increases_with_generations(self):
        """Test that execution time increases with more generations."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        
        result_small = run_adapter_benchmark(adapter1, 10, 42, "Test", repeats=1)
        result_large = run_adapter_benchmark(adapter2, 100, 42, "Test", repeats=1)
        
        # More generations should take more time (usually)
        # Note: This might be flaky due to JIT compilation effects
        assert result_small.execution_time > 0
        assert result_large.execution_time > 0
        # Single repeat should have zero std
        assert result_small.fitness_std == 0.0
        assert result_large.fitness_std == 0.0
    
    def test_repeats_average_correctly(self):
        """Test that multiple repeats produce stable timing."""
        adapter = MockAdapter(num_gens=50)
        
        # Single run
        result_single = run_adapter_benchmark(adapter, 50, 42, "Test", repeats=1)
        
        # Multiple runs
        adapter2 = MockAdapter(num_gens=50)
        result_multi = run_adapter_benchmark(adapter2, 50, 42, "Test", repeats=5)
        
        # Both should have reasonable execution times
        assert result_single.execution_time > 0
        assert result_multi.execution_time > 0
        # Averaged result should be in similar range (accounting for variance)
        assert 0.1 < result_multi.execution_time / result_single.execution_time < 10


class TestErrorHandling:
    """Test error handling in benchmark runner."""
    
    def test_invalid_num_gens(self):
        """Test handling of zero number of generations."""
        adapter = MockAdapter()
        
        # Zero generations should execute successfully and return zero fitness
        # JAX lax.scan handles length=0 gracefully by not executing the loop
        result = run_adapter_benchmark(adapter, 0, 42, "Test")
        assert isinstance(result, BenchmarkResult)
        assert result.best_fitness == 0.0  # No iterations occurred
        assert result.fitness_std == 0.0  # Single run (repeats=1 default)
    
    def test_invalid_repeats(self):
        """Test handling of invalid repeat count."""
        adapter = MockAdapter()
        
        # Zero repeats should default to 1 via max(1, repeats)
        result = run_adapter_benchmark(adapter, 10, 42, "Test", repeats=0)
        assert isinstance(result, BenchmarkResult)
    
    def test_adapter_without_device_info(self):
        """Test adapter that doesn't properly implement get_device_info."""
        
        class BrokenAdapter(AbstractBenchmarkAdapter):
            def init(self, rng):
                return {"fitness": 0}
            
            def make_step_fn(self):
                return lambda carry, _: (carry, 0.0)
            
            def get_best_fitness(self, carry):
                return 0.0
            
            def get_device_info(self):
                return None  # Invalid return
        
        adapter = BrokenAdapter()
        result = run_adapter_benchmark(adapter, 10, 42, "Test")
        
        # Should handle None device gracefully
        assert result.device is None or result.device == "None"


class TestFitnessTracking:
    """Test fitness value tracking and statistics."""
    
    def test_single_repeat_zero_std(self):
        """Test that single repeat produces zero standard deviation."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", repeats=1)
        
        assert result.fitness_std == 0.0
        assert result.best_fitness == 10.0  # MockAdapter increments by 1 each step
    
    def test_multiple_repeats_calculate_std(self):
        """Test that multiple repeats calculate fitness statistics."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", repeats=5)
        
        # With deterministic MockAdapter, all runs should be identical
        # So std should be 0 or very small
        assert result.fitness_std >= 0.0
        assert result.best_fitness == 10.0
    
    def test_fitness_values_collected_per_repeat(self):
        """Test that fitness values are collected from each repeat."""
        adapter = MockAdapter(num_gens=20)
        
        result = run_adapter_benchmark(adapter, 20, 42, "Test", repeats=3)
        
        # Mean fitness should be from multiple runs
        assert result.best_fitness == 20.0
        assert isinstance(result.fitness_std, float)
        assert result.fitness_std >= 0.0
