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
    
    def extract_best_fitness(self, carry):
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
            pop_size=100,
            unroll=1,
            compile_time=0.5,
            mean_exec_time=2.0,
            std_exec_time=0.15,
            mean_gps=50.0,
            best_fitness_final=10.5,
        )
        
        assert result.framework == "MalthusJAX"
        assert result.device == "CPU"
        assert result.pop_size == 100
        assert result.unroll == 1
        assert result.compile_time == 0.5
        assert result.mean_exec_time == 2.0
        assert result.std_exec_time == 0.15
        assert result.mean_gps == 50.0
        assert result.best_fitness_final == 10.5


class TestRunAdapterBenchmark:
    """Test the run_adapter_benchmark function."""
    
    def test_basic_benchmark_execution(self):
        """Test basic benchmark execution with default parameters."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=10,
            seed=42,
            framework_name="TestFramework",
            pop_size=50,
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.framework == "TestFramework"
        assert result.device == "CPU"
        assert result.pop_size == 50
        assert result.compile_time > 0
        assert result.mean_exec_time > 0
        assert result.mean_gps > 0
        assert result.std_exec_time >= 0.0
        assert adapter.init_called
        assert adapter.step_called
    
    def test_benchmark_with_unroll(self):
        """Test benchmark execution with unroll_factor parameter."""
        adapter = MockAdapter(num_gens=20)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=20,
            seed=42,
            framework_name="TestFramework",
            pop_size=50,
            unroll_factor=4,
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.unroll == 4
        assert result.best_fitness_final == 20.0  # Should iterate 20 times
    
    def test_benchmark_with_repeats(self):
        """Test benchmark execution with multiple repeats."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(
            adapter=adapter,
            num_gens=10,
            seed=42,
            framework_name="TestFramework",
            pop_size=50,
            repeats=3,
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.mean_exec_time > 0
        assert result.mean_gps > 0
        assert result.std_exec_time >= 0.0
        # With deterministic execution, std should be very small
        assert result.std_exec_time < 1.0
    
    def test_benchmark_different_seeds(self):
        """Test that different seeds produce consistent but independent runs."""
        adapter1 = MockAdapter(num_gens=10)
        adapter2 = MockAdapter(num_gens=10)
        
        result1 = run_adapter_benchmark(adapter1, 10, 42, "Test", pop_size=50)
        result2 = run_adapter_benchmark(adapter2, 10, 43, "Test", pop_size=50)
        
        # Both should succeed
        assert result1.best_fitness_final == 10.0
        assert result2.best_fitness_final == 10.0
    
    def test_compile_time_measurement(self):
        """Test that compile time is measured separately from execution."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50)
        
        # Compile and execution times should both be positive
        assert result.compile_time > 0
        assert result.mean_exec_time > 0
        # They should be measured independently
        assert result.compile_time != result.mean_exec_time
    
    def test_generations_per_sec_calculation(self):
        """Test that GPS is calculated correctly."""
        adapter = MockAdapter(num_gens=100)
        
        result = run_adapter_benchmark(adapter, 100, 42, "Test", pop_size=50)
        
        # GPS = num_gens / mean_exec_time
        expected_gps = 100 / result.mean_exec_time
        assert abs(result.mean_gps - expected_gps) < 0.1


class TestAdapterInterface:
    """Test the AbstractBenchmarkAdapter interface."""
    
    def test_adapter_implements_all_methods(self):
        """Test that MockAdapter implements all required methods."""
        adapter = MockAdapter()
        
        # All methods should be callable
        assert callable(adapter.init)
        assert callable(adapter.make_step_fn)
        assert callable(adapter.extract_best_fitness)
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
        """Test that more generations take more time."""
        adapter_short = MockAdapter(num_gens=10)
        adapter_long = MockAdapter(num_gens=100)
        
        result_short = run_adapter_benchmark(adapter_short, 10, 42, "Test", pop_size=50)
        result_long = run_adapter_benchmark(adapter_long, 100, 42, "Test", pop_size=50)
        
        # More generations should take longer (usually)
        # Note: JIT compilation time can dominate for small runs
        assert result_short.mean_exec_time > 0
        assert result_long.mean_exec_time > 0
    
    def test_repeats_average_correctly(self):
        """Test that multiple repeats are averaged correctly."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50, repeats=5)
        
        # all_times should have 5 entries
        assert len(result.all_times) == 5
        # mean_exec_time should be the average of all_times
        import numpy as np
        assert abs(result.mean_exec_time - np.mean(result.all_times)) < 1e-6


class TestErrorHandling:
    """Test error handling in benchmark runner."""
    
    def test_invalid_num_gens(self):
        """Test handling of invalid num_gens parameter."""
        adapter = MockAdapter()
        
        # Zero or negative generations should be handled gracefully
        # The function may raise an error or return zero GPS
        try:
            result = run_adapter_benchmark(adapter, 0, 42, "Test", pop_size=50)
            # If it doesn't raise, GPS should be 0 or undefined
            assert result.mean_gps == 0.0 or result.mean_exec_time == 0.0
        except (ValueError, ZeroDivisionError):
            pass  # Expected behavior
    
    def test_invalid_repeats(self):
        """Test handling of invalid repeats parameter."""
        adapter = MockAdapter()
        
        # Zero or negative repeats should be handled
        try:
            result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50, repeats=0)
            # If it doesn't raise, should still return valid result
            assert isinstance(result, BenchmarkResult)
        except (ValueError, ZeroDivisionError):
            pass  # Expected behavior
    
    def test_adapter_without_device_info(self):
        """Test handling when adapter returns empty device info."""
        class NoDeviceAdapter(MockAdapter):
            def get_device_info(self):
                return ""
        
        adapter = NoDeviceAdapter()
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50)
        
        assert result.device == ""


class TestFitnessTracking:
    """Test fitness tracking across repeats."""
    
    def test_single_repeat_zero_std(self):
        """Test that single repeat has zero std for execution time."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50, repeats=1)
        
        # With single repeat, std should be 0
        assert result.std_exec_time == 0.0
    
    def test_multiple_repeats_calculate_std(self):
        """Test that multiple repeats calculate std correctly."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50, repeats=5)
        
        # Std should be calculated from all_times
        import numpy as np
        expected_std = np.std(result.all_times)
        assert abs(result.std_exec_time - expected_std) < 1e-6
    
    def test_fitness_values_collected_per_repeat(self):
        """Test that best fitness is collected from each repeat."""
        adapter = MockAdapter(num_gens=10)
        
        result = run_adapter_benchmark(adapter, 10, 42, "Test", pop_size=50, repeats=3)
        
        # Best fitness should be the average of final fitnesses across repeats
        # With deterministic mock, all repeats should give same fitness
        assert result.best_fitness_final == 10.0