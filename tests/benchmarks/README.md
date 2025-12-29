# Benchmarking Test Suite

This directory contains comprehensive tests for the MalthusJAX benchmarking framework.

## Test Structure

### `test_cli.py`
Tests for the command-line interface and main orchestration logic:
- **TestLoadConfig**: TOML configuration loading and validation
- **TestSetupBBOBInstances**: BBOB problem instantiation 
- **TestRunSingleBenchmark**: Individual benchmark execution
- **TestMainIntegration**: End-to-end CLI integration tests
- **TestMemoryManagement**: Memory cleanup verification

### `test_runner.py`
Tests for the core benchmark runner:
- **TestBenchmarkResult**: Result dataclass validation
- **TestRunAdapterBenchmark**: Benchmark execution with various parameters
- **TestAdapterInterface**: Adapter interface compliance
- **TestPerformanceMeasurement**: Timing and performance metrics
- **TestErrorHandling**: Edge cases and error conditions

### `test_adapters.py`
Tests for framework adapters (MalthusJAX and Evosax):
- **TestAbstractBenchmarkAdapter**: Abstract base class interface
- **TestMalthusAdapter**: MalthusJAX adapter implementation
- **TestEvosaxAdapter**: Evosax adapter implementation
- **TestAdapterComparison**: Cross-adapter consistency

## Running Tests

Run all benchmark tests:
```bash
pytest tests/benchmarks/ -v
```

Run specific test file:
```bash
pytest tests/benchmarks/test_cli.py -v
```

Run specific test class:
```bash
pytest tests/benchmarks/test_runner.py::TestRunAdapterBenchmark -v
```

Run with coverage:
```bash
pytest tests/benchmarks/ --cov=benchmarks --cov-report=html
```

## Test Features

- **Mocking**: Extensive use of mocks to isolate units and avoid expensive computations
- **Integration tests**: End-to-end tests with temporary files and cleanup
- **Parameter sweep tests**: Validates grid search over all parameter combinations
- **Memory management**: Ensures proper cleanup with `jax.clear_caches()`
- **Performance validation**: Checks timing measurements and GPS calculations

## Test Data

Tests use minimal configurations to run quickly:
- Small population sizes (16-32)
- Few generations (5-10)
- Low-dimensional problems (5-10D)
- Single/few seeds

## Fixtures

Common test fixtures include:
- `mock_spec`: Mocked ComparisonSpec from registry
- `mock_engine`: Mocked GeneticEngine
- `mock_strategy`: Mocked Evosax strategy
- `mock_problem`: Mocked BBOB problem
- `minimal_config`: Minimal valid TOML configuration

## Notes

- Some tests create temporary directories (`results/test_*`) which are cleaned up automatically
- JAX compilation can cause timing tests to be variable; use appropriate tolerances
- Integration tests may require the full MalthusJAX package to be installed
