import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import jax.numpy as jnp

from benchmarks.cli import (
    load_config,
    setup_bbob_instances,
    run_single_benchmark,
    main,
)
from benchmarks.framework.runner import BenchmarkResult


class TestLoadConfig:
    """Test TOML configuration loading."""

    def test_load_valid_toml(self, tmp_path):
        """Test loading a valid TOML configuration file."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[experi                            ment]
name = "Test_Benchmark"
output_dir = "results/test"
repeats = 2

[grid]
algorithms = ["Standard_GA"]
tasks = ["sphere"]
dimensions = [5]
pop_sizes = [32]
seeds = [42]
generations = 10
unroll_factors = [1]
""")
        
        cfg = load_config(str(config_file))
        
        assert cfg['experiment']['name'] == "Test_Benchmark"
        assert cfg['experiment']['repeats'] == 2
        assert cfg['grid']['algorithms'] == ["Standard_GA"]
        assert cfg['grid']['pop_sizes'] == [32]

    def test_load_missing_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.toml")


class TestSetupBBOBInstances:
    """Test BBOB problem instantiation."""

    def test_setup_sphere_problem(self):
        """Test setting up a Sphere problem instance."""
        m_eval, e_prob = setup_bbob_instances("sphere", dim=5, seed=42)
        
        assert m_eval is not None
        assert e_prob is not None
        assert hasattr(e_prob, 'num_dims')
        assert e_prob.num_dims == 5

    def test_setup_rastrigin_problem(self):
        """Test setting up a Rastrigin problem instance."""
        m_eval, e_prob = setup_bbob_instances("rastrigin", dim=10, seed=123)
        
        assert m_eval is not None
        assert e_prob is not None
        assert e_prob.num_dims == 10

    def test_different_seeds_create_different_instances(self):
        """Test that different seeds create different problem instances."""
        m1, e1 = setup_bbob_instances("sphere", dim=5, seed=42)
        m2, e2 = setup_bbob_instances("sphere", dim=5, seed=43)
        
        # Instances should be different objects
        assert m1 is not m2
        assert e1 is not e2


class TestRunSingleBenchmark:
    """Test individual benchmark execution."""

    @pytest.fixture
    def mock_spec(self):
        """Create a mock ComparisonSpec."""
        spec = Mock()
        spec.malthus_factory = Mock(return_value=Mock())
        spec.evosax_factory = Mock(return_value=Mock())
        return spec

    @pytest.fixture
    def mock_benchmark_result(self):
        """Create a mock BenchmarkResult."""
        return BenchmarkResult(
            framework="TestFramework",
            device="CPU",
            compile_time=0.1,
            execution_time=0.5,
            generations_per_sec=20.0,
            best_fitness=10.0,
        )

    @patch('benchmarks.cli.setup_bbob_instances')
    @patch('benchmarks.cli.run_adapter_benchmark')
    @patch('jax.clear_caches')
    def test_run_single_benchmark_returns_records(
        self,
        mock_clear_caches,
        mock_run_adapter,
        mock_setup,
        mock_spec,
        mock_benchmark_result,
    ):
        """Test that run_single_benchmark returns properly formatted records."""
        # Setup mocks
        mock_setup.return_value = (Mock(), Mock())
        mock_run_adapter.return_value = mock_benchmark_result
        
        # Run benchmark
        rec_m, rec_e = run_single_benchmark(
            algo_name="Standard_GA",
            spec=mock_spec,
            hypers={"mutation_rate": 0.1},
            task="sphere",
            dim=5,
            pop_size=32,
            seed=42,
            unroll=1,
            generations=10,
            repeats=1,
            run_num=1,
            total_runs=1,
        )
        
        # Verify records structure
        assert rec_m["Algorithm"] == "Standard_GA"
        assert rec_m["Task"] == "sphere"
        assert rec_m["Dim"] == 5
        assert rec_m["Pop_Size"] == 32
        assert rec_m["Seed"] == 42
        assert rec_m["Unroll"] == 1
        assert rec_m["Framework"] == "MalthusJAX"
        
        assert rec_e["Framework"] == "Evosax"
        assert rec_e["Algorithm"] == "Standard_GA"
        
        # Verify cleanup was called
        mock_clear_caches.assert_called_once()

    @patch('benchmarks.cli.setup_bbob_instances')
    @patch('benchmarks.cli.run_adapter_benchmark')
    def test_run_single_benchmark_calls_adapters(
        self,
        mock_run_adapter,
        mock_setup,
        mock_spec,
        mock_benchmark_result,
    ):
        """Test that adapters are called with correct parameters."""
        mock_setup.return_value = (Mock(), Mock())
        mock_run_adapter.return_value = mock_benchmark_result
        
        run_single_benchmark(
            algo_name="Standard_GA",
            spec=mock_spec,
            hypers={"mutation_rate": 0.1},
            task="sphere",
            dim=5,
            pop_size=32,
            seed=42,
            unroll=2,
            generations=100,
            repeats=3,
            run_num=1,
            total_runs=1,
        )
        
        # Verify factories were called
        mock_spec.malthus_factory.assert_called_once()
        mock_spec.evosax_factory.assert_called_once()
        
        # Verify run_adapter_benchmark was called with unroll and repeats
        assert mock_run_adapter.call_count == 2
        call_kwargs = mock_run_adapter.call_args_list[0][1]
        assert call_kwargs['unroll'] == 2
        assert call_kwargs['repeats'] == 3


class TestMainIntegration:
    """Integration tests for the main CLI function."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal valid configuration file."""
        config_file = tmp_path / "minimal_config.toml"
        config_file.write_text("""
[experiment]
name = "Minimal_Test"
output_dir = "results/test_minimal"
repeats = 1

[grid]
algorithms = ["Standard_GA"]
tasks = ["sphere"]
dimensions = [5]
pop_sizes = [16]
seeds = [42]
generations = 5
unroll_factors = [1]

[grid.hyperparams]
mutation_rate = 0.05
crossover_rate = 0.6
sigma = 0.1
elite_ratio = 0.1
""")
        return str(config_file)

    @patch('benchmarks.cli.run_single_benchmark')
    @patch('sys.argv', ['cli.py', 'dummy_config.toml'])
    def test_main_creates_output_directory(self, mock_run_single, minimal_config, tmp_path):
        """Test that main creates the output directory."""
        mock_run_single.return_value = (
            {
                "Algorithm": "Standard_GA",
                "Task": "sphere",
                "Dim": 5,
                "Pop_Size": 16,
                "Generations": 5,
                "Seed": 42,
                "Unroll": 1,
                "Device": "CPU",
                "Framework": "MalthusJAX",
                "Compile_Time": 0.1,
                "Exec_Time": 0.5,
                "GPS": 10.0,
                "Best_Fitness": 5.0,
                "Fitness_Std": 0.05,
            },
            {
                "Algorithm": "Standard_GA",
                "Task": "sphere",
                "Dim": 5,
                "Pop_Size": 16,
                "Generations": 5,
                "Seed": 42,
                "Unroll": 1,
                "Device": "CPU",
                "Framework": "Evosax",
                "Compile_Time": 0.1,
                "Exec_Time": 0.6,
                "GPS": 8.3,
                "Best_Fitness": 5.5,
                "Fitness_Std": 0.08,
            },
        )
        
        with patch('sys.argv', ['cli.py', minimal_config]):
            with patch('jax.devices', return_value=[Mock(device_kind="CPU")]):
                main()
        
        # Verify output directory was created
        assert os.path.exists("results/test_minimal")
        
        # Cleanup
        import shutil
        if os.path.exists("results/test_minimal"):
            shutil.rmtree("results/test_minimal")

    @patch('benchmarks.cli.run_single_benchmark')
    @patch('sys.argv', ['cli.py', 'dummy_config.toml'])
    def test_main_saves_csv_output(self, mock_run_single, minimal_config, tmp_path):
        """Test that main saves results to CSV."""
        mock_run_single.return_value = (
            {
                "Algorithm": "Standard_GA",
                "Task": "sphere",
                "Dim": 5,
                "Pop_Size": 16,
                "Generations": 5,
                "Seed": 42,
                "Unroll": 1,
                "Device": "CPU",
                "Framework": "MalthusJAX",
                "Compile_Time": 0.1,
                "Exec_Time": 0.5,
                "GPS": 10.0,
                "Best_Fitness": 5.0,
                "Fitness_Std": 0.05,
            },
            {
                "Algorithm": "Standard_GA",
                "Task": "sphere",
                "Dim": 5,
                "Pop_Size": 16,
                "Generations": 5,
                "Seed": 42,
                "Unroll": 1,
                "Device": "CPU",
                "Framework": "Evosax",
                "Compile_Time": 0.1,
                "Exec_Time": 0.6,
                "GPS": 8.3,
                "Best_Fitness": 5.5,
                "Fitness_Std": 0.08,
            },
        )
        
        with patch('sys.argv', ['cli.py', minimal_config]):
            with patch('jax.devices', return_value=[Mock(device_kind="CPU")]):
                main()
        
        # Verify CSV file was created
        output_files = list(Path("results/test_minimal").glob("benchmark_*.csv"))
        assert len(output_files) > 0
        
        # Verify CSV contents
        df = pd.read_csv(output_files[0])
        assert len(df) == 2  # One row for MalthusJAX, one for Evosax
        assert "Framework" in df.columns
        assert "GPS" in df.columns
        assert "Best_Fitness" in df.columns
        assert "Fitness_Std" in df.columns
        
        # Cleanup
        import shutil
        if os.path.exists("results/test_minimal"):
            shutil.rmtree("results/test_minimal")

    def test_main_calculates_correct_total_runs(self, minimal_config, tmp_path):
        """Test that total_runs is calculated correctly for parameter grid."""
        with patch('benchmarks.cli.run_single_benchmark') as mock_run:
            mock_run.return_value = (
                {"Algorithm": "GA", "Framework": "MalthusJAX", "GPS": 10, "Best_Fitness": 5.0, "Fitness_Std": 0.1},
                {"Algorithm": "GA", "Framework": "Evosax", "GPS": 8, "Best_Fitness": 5.2, "Fitness_Std": 0.15},
            )
            
            # Create config with multiple parameters
            multi_config = tmp_path / "multi_config.toml"
            multi_config.write_text("""
[experiment]
name = "Multi_Test"
output_dir = "results/test_multi"
repeats = 1

[grid]
algorithms = ["Standard_GA"]
tasks = ["sphere", "rastrigin"]
dimensions = [5, 10]
pop_sizes = [16, 32]
seeds = [42, 43]
generations = 5
unroll_factors = [1, 2]
""")
            
            with patch('sys.argv', ['cli.py', str(multi_config)]):
                with patch('jax.devices', return_value=[Mock(device_kind="CPU")]):
                    main()
            
            # Expected: 1 algo * 2 tasks * 2 dims * 2 pop_sizes * 2 seeds * 2 unrolls = 32 calls
            assert mock_run.call_count == 32
            
            # Cleanup
            import shutil
            if os.path.exists("results/test_multi"):
                shutil.rmtree("results/test_multi")

    def test_main_passes_normalize_flag(self, tmp_path):
        """Test that the normalize_fitness option from the TOML is passed to run_single_benchmark."""
        # Create a config that sets normalize_fitness
        cfg = tmp_path / "norm_config.toml"
        cfg.write_text('''
[experiment]
name = "Norm_Test"
output_dir = "results/test_norm"
repeats = 1
normalize_fitness = "malthus"

[grid]
algorithms = ["Standard_GA"]
tasks = ["sphere"]
dimensions = [5]
pop_sizes = [16]
seeds = [42]
generations = 5
unroll_factors = [1]
''')

        with patch('benchmarks.cli.run_single_benchmark') as mock_run:
            mock_run.return_value = (
                {"Algorithm": "Standard_GA", "Framework": "MalthusJAX", "GPS": 10, "Best_Fitness": 0.0, "Fitness_Std": 0.0},
                {"Algorithm": "Standard_GA", "Framework": "Evosax", "GPS": 8, "Best_Fitness": 0.0, "Fitness_Std": 0.0},
            )
            with patch('sys.argv', ['cli.py', str(cfg)]):
                with patch('jax.devices', return_value=[Mock(device_kind="CPU")]):
                    main()

        # Ensure run_single_benchmark was called and that normalize kwarg was passed
        assert mock_run.call_count >= 1
        called_kwargs = mock_run.call_args_list[0][1]
        assert 'normalize' in called_kwargs
        assert called_kwargs['normalize'] == 'malthus'

        # Cleanup
        import shutil
        if os.path.exists("results/test_norm"):
            shutil.rmtree("results/test_norm")


class TestMemoryManagement:
    """Test memory cleanup and resource management."""

    @patch('benchmarks.cli.setup_bbob_instances')
    @patch('benchmarks.cli.run_adapter_benchmark')
    @patch('jax.clear_caches')
    def test_memory_cleanup_called(
        self,
        mock_clear_caches,
        mock_run_adapter,
        mock_setup,
    ):
        """Test that JAX cache clearing is called after each benchmark."""
        mock_setup.return_value = (Mock(), Mock())
        mock_run_adapter.return_value = BenchmarkResult(
            framework="Test",
            device="CPU",
            compile_time=0.1,
            execution_time=0.5,
            generations_per_sec=20.0,
            best_fitness=10.0,
            fitness_std=0.05,
        )
        
        spec = Mock()
        spec.malthus_factory = Mock(return_value=Mock())
        spec.evosax_factory = Mock(return_value=Mock())
        
        run_single_benchmark(
            algo_name="Standard_GA",
            spec=spec,
            hypers={},
            task="sphere",
            dim=5,
            pop_size=32,
            seed=42,
            unroll=1,
            generations=10,
            repeats=1,
            run_num=1,
            total_runs=1,
        )
        
        # Verify cache was cleared
        mock_clear_caches.assert_called_once()
