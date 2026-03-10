import os
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from benchmarks.cli import (
    load_config,
    main,
    run_single_benchmark,
    setup_bbob_instances,
)
from benchmarks.framework.runner import BenchmarkResult


class TestLoadConfig:
    """Test TOML configuration loading."""

    def test_load_valid_toml(self, tmp_path):
        """Test loading a valid TOML configuration file."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[experiment]
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

        assert cfg["experiment"]["name"] == "Test_Benchmark"
        assert cfg["experiment"]["repeats"] == 2
        assert cfg["grid"]["algorithms"] == ["Standard_GA"]
        assert cfg["grid"]["pop_sizes"] == [32]

    def test_load_missing_file(self):
        """Test loading a non-existent file raises error.

        Accepts both FileNotFoundError and PermissionError: on Linux, opening a
        path whose parent directory doesn't exist at the root level raises
        PermissionError (EACCES) rather than FileNotFoundError (ENOENT).
        """
        with pytest.raises((FileNotFoundError, PermissionError)):
            load_config("/nonexistent/path/config.toml")


class TestSetupBBOBInstances:
    """Test BBOB problem instantiation."""

    def test_setup_sphere_problem(self):
        """Test setting up a Sphere problem instance."""
        m_eval, e_prob = setup_bbob_instances("sphere", dim=5, seed=42)

        assert m_eval is not None
        assert e_prob is not None
        assert hasattr(e_prob, "num_dims")
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
            pop_size=32,
            unroll=1,
            compile_time=0.1,
            mean_exec_time=0.5,
            std_exec_time=0.05,
            mean_gps=20.0,
            best_fitness_final=10.0,
            fitness_std=0.0,
        )

    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    @patch("jax.clear_caches")
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

    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
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
        # run_adapter_benchmark is called with positional args:
        # (adapter, generations, seed, framework_name, pop_size, unroll_factor, repeats)
        call_args = mock_run_adapter.call_args_list[0][0]  # positional args
        # args[5] = unroll_factor, args[6] = repeats
        assert call_args[5] == 2  # unroll
        assert call_args[6] == 3  # repeats


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

    @pytest.fixture
    def mock_benchmark_result_m(self):
        """Mock MalthusJAX benchmark result."""
        return BenchmarkResult(
            framework="MalthusJAX",
            device="CPU",
            pop_size=16,
            unroll=1,
            compile_time=0.1,
            mean_exec_time=0.5,
            std_exec_time=0.05,
            mean_gps=10.0,
            best_fitness_final=5.0,
            fitness_std=0.0,
            all_times=[],
        )

    @pytest.fixture
    def mock_benchmark_result_e(self):
        """Mock Evosax benchmark result."""
        return BenchmarkResult(
            framework="Evosax",
            device="CPU",
            pop_size=16,
            unroll=1,
            compile_time=0.1,
            mean_exec_time=0.6,
            std_exec_time=0.08,
            mean_gps=8.3,
            best_fitness_final=5.5,
            fitness_std=0.0,
            all_times=[],
        )

    @patch("benchmarks.cli.ComparisonRegistry")
    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    def test_main_creates_output_directory(
        self,
        mock_run_adapter,
        mock_setup,
        mock_registry,
        minimal_config,
        mock_benchmark_result_m,
        mock_benchmark_result_e,
        tmp_path,
    ):
        """Test that main creates the output directory."""
        mock_setup.return_value = (Mock(), Mock())
        # Mock the registry to return a spec with mocked factories
        mock_spec = Mock()
        mock_spec.default_hypers = {}
        mock_spec.malthus_factory = Mock(return_value=Mock())
        mock_spec.evosax_factory = Mock(return_value=Mock())
        mock_registry.get.return_value = mock_spec
        # Return MalthusJAX result first, then Evosax result
        mock_run_adapter.side_effect = [mock_benchmark_result_m, mock_benchmark_result_e]

        with patch("sys.argv", ["cli.py", minimal_config]):
            with patch("jax.devices", return_value=[Mock(device_kind="CPU")]):
                main()

        # Verify output directory was created
        assert os.path.exists("results/test_minimal")

        # Cleanup
        import shutil

        if os.path.exists("results/test_minimal"):
            shutil.rmtree("results/test_minimal")

    @patch("benchmarks.cli.ComparisonRegistry")
    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    def test_main_saves_csv_output(
        self,
        mock_run_adapter,
        mock_setup,
        mock_registry,
        minimal_config,
        mock_benchmark_result_m,
        mock_benchmark_result_e,
        tmp_path,
    ):
        """Test that main saves results to CSV."""
        mock_setup.return_value = (Mock(), Mock())
        mock_spec = Mock()
        mock_spec.default_hypers = {}
        mock_spec.malthus_factory = Mock(return_value=Mock())
        mock_spec.evosax_factory = Mock(return_value=Mock())
        mock_registry.get.return_value = mock_spec
        mock_run_adapter.side_effect = [mock_benchmark_result_m, mock_benchmark_result_e]

        with patch("sys.argv", ["cli.py", minimal_config]):
            with patch("jax.devices", return_value=[Mock(device_kind="CPU")]):
                main()

        # Verify CSV file was created (main() creates final_benchmark_{timestamp}.csv)
        output_files = list(Path("results/test_minimal").glob("final_benchmark_*.csv"))
        assert len(output_files) > 0

        # Verify CSV contents (column names match what main() produces)
        df = pd.read_csv(output_files[0])
        assert len(df) == 2  # One row for MalthusJAX, one for Evosax
        assert "Framework" in df.columns
        assert "Mean_GPS" in df.columns
        assert "Best_Fitness" in df.columns

        # Cleanup
        import shutil

        if os.path.exists("results/test_minimal"):
            shutil.rmtree("results/test_minimal")

    @patch("benchmarks.cli.ComparisonRegistry")
    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    def test_main_calculates_correct_total_runs(
        self,
        mock_run_adapter,
        mock_setup,
        mock_registry,
        minimal_config,
        mock_benchmark_result_m,
        mock_benchmark_result_e,
        tmp_path,
    ):
        """Test that total_runs is calculated correctly for parameter grid."""
        mock_setup.return_value = (Mock(), Mock())
        # Mock the registry
        mock_spec = Mock()
        mock_spec.default_hypers = {}
        mock_spec.malthus_factory = Mock(return_value=Mock())
        mock_spec.evosax_factory = Mock(return_value=Mock())
        mock_registry.get.return_value = mock_spec

        # Create results that will be returned for each call
        def make_result_m():
            return BenchmarkResult(
                framework="MalthusJAX",
                device="CPU",
                pop_size=16,
                unroll=1,
                compile_time=0.1,
                mean_exec_time=0.5,
                std_exec_time=0.05,
                mean_gps=10.0,
                best_fitness_final=5.0,
                fitness_std=0.0,
                all_times=[],
            )

        def make_result_e():
            return BenchmarkResult(
                framework="Evosax",
                device="CPU",
                pop_size=16,
                unroll=1,
                compile_time=0.1,
                mean_exec_time=0.6,
                std_exec_time=0.08,
                mean_gps=8.0,
                best_fitness_final=5.2,
                fitness_std=0.0,
                all_times=[],
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
seeds = [42]
generations = 5
unroll_factors = [1, 2]
""")
        # main() iterates: algo x task x dim x pop x unroll = 1*2*2*2*2 = 16 configs
        # Each config calls run_adapter_benchmark twice (MalthusJAX + Evosax)
        # So total calls = 16 * 2 = 32
        mock_run_adapter.side_effect = [
            make_result_m() if i % 2 == 0 else make_result_e() for i in range(32)
        ]

        with patch("sys.argv", ["cli.py", str(multi_config)]):
            with patch("jax.devices", return_value=[Mock(device_kind="CPU")]):
                main()

        # Verify run_adapter_benchmark was called 32 times (16 configs * 2 frameworks)
        assert mock_run_adapter.call_count == 32

        # Cleanup
        import shutil

        if os.path.exists("results/test_multi"):
            shutil.rmtree("results/test_multi")

    @patch("benchmarks.cli.ComparisonRegistry")
    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    def test_main_passes_normalize_flag(
        self, mock_run_adapter, mock_setup, mock_registry, tmp_path
    ):
        """
        Test that normalize_fitness option is handled (placeholder - main() doesn't use it yet).
        """
        mock_setup.return_value = (Mock(), Mock())
        mock_spec = Mock()
        mock_spec.default_hypers = {}
        mock_spec.malthus_factory = Mock(return_value=Mock())
        mock_spec.evosax_factory = Mock(return_value=Mock())
        mock_registry.get.return_value = mock_spec
        mock_run_adapter.side_effect = [
            BenchmarkResult(
                framework="MalthusJAX",
                device="CPU",
                pop_size=16,
                unroll=1,
                compile_time=0.1,
                mean_exec_time=0.5,
                std_exec_time=0.0,
                mean_gps=10.0,
                best_fitness_final=0.0,
                fitness_std=0.0,
                all_times=[],
            ),
            BenchmarkResult(
                framework="Evosax",
                device="CPU",
                pop_size=16,
                unroll=1,
                compile_time=0.1,
                mean_exec_time=0.6,
                std_exec_time=0.0,
                mean_gps=8.0,
                best_fitness_final=0.0,
                fitness_std=0.0,
                all_times=[],
            ),
        ]

        # Create a config that sets normalize_fitness
        cfg = tmp_path / "norm_config.toml"
        cfg.write_text("""
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
""")

        with patch("sys.argv", ["cli.py", str(cfg)]):
            with patch("jax.devices", return_value=[Mock(device_kind="CPU")]):
                main()

        # Verify benchmark was called (normalize_fitness is a config option,
        # main() currently doesn't pass it to run_adapter_benchmark)
        assert mock_run_adapter.call_count == 2

        # Cleanup
        import shutil

        if os.path.exists("results/test_norm"):
            shutil.rmtree("results/test_norm")


class TestMemoryManagement:
    """Test memory cleanup and resource management."""

    @patch("benchmarks.cli.setup_bbob_instances")
    @patch("benchmarks.cli.run_adapter_benchmark")
    @patch("jax.clear_caches")
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
            pop_size=32,
            unroll=1,
            compile_time=0.1,
            mean_exec_time=0.5,
            std_exec_time=0.05,
            mean_gps=20.0,
            best_fitness_final=10.0,
            fitness_std=0.0,
            all_times=[],
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
