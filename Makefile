.PHONY: help install-dev install-bench test test-fast test-failing test-fixes test-bench test-bench-snapshot \
        lint format format-check type-check check-all docs bench

help:
	@echo "--- MalthusJAX Development ---"
	@echo "  make install-dev    Install package + dev/docs/examples deps"
	@echo "  make install-bench  Install package + benchmark deps (evosax, pandas, scipy)"
	@echo "  make test           Run full pytest suite with coverage (min 80%)"
	@echo "  make test-fast      Run full suite, skip coverage (faster iteration)"
	@echo "  make test-failing       Re-run only the subset known to fail on multi-GPU hosts"
	@echo "  make test-fixes         Re-run the two most-recently fixed tests (cli + bfloat16)"
	@echo "  make test-bench         Run functional tests in tests/benchmarks/ (no timing)"
	@echo "  make test-bench-snapshot  Run pytest-benchmark snapshot suite"
	@echo "  make bench              Run TOML-driven dispatch timing CLI benchmark"
	@echo "  make *-nohup           Run the same command under nohup and log output to a timestamped file"
	@echo "  make lint               Ruff lint check (no fixes)"
	@echo "  make format             Ruff auto-format (mutates files)"
	@echo "  make format-check       Ruff format check only (no mutations)"
	@echo "  make type-check         mypy strict check on src/"
	@echo "  make check-all          lint + format-check + type-check + test"
	@echo "  make docs               Build Sphinx HTML docs"

install-dev:
	@echo "--- Installing dev dependencies ---"
	pip install -e ".[dev,docs,examples]"

install-bench:
	@echo "--- Installing benchmark dependencies ---"
	pip install -e ".[dev,benchmarks]"

test:
	@echo "--- Running tests with coverage ---"
	python -m pytest

test-fast:
	@echo "--- Running tests without coverage ---"
	python -m pytest --no-cov -q

test-failing:
	@echo "--- Running previously failing test subset ---"
	python -m pytest --no-cov -q \
	  tests/composer/test_engine_factory.py::test_engine_adapter_run_once \
	  tests/composer/test_optimization_direction.py::TestOptimizationDirection::test_optimization_directions_are_opposite \
	  tests/engine/test_engine_edge_cases_fixed.py \
	  tests/engine/test_engine_quality_fixed.py \
	  tests/engine/test_genetic_engine.py \
	  tests/engine/test_genetic_engine_asktel.py \
	  tests/engine/test_genetic_engine_core.py \
	  tests/engine/test_genetic_engine_fixes.py \
	  tests/engine/test_genetic_engine_jit.py \
	  tests/engine/test_genetic_engine_scheduling.py \
	  tests/engine/test_genetic_engine_phases.py::TestSelectionPhase::test_elite_genes_are_best_fitness \
	  tests/engine/test_hof_tracking.py::TestTrackBestLight::test_history_is_monotonic \
	  tests/engine/test_hof_tracking.py::TestTrackBestFull::test_best_fitness_tracks_global_best \
	  tests/engine/test_hof_tracking.py::TestTrackBestFull::test_history_is_monotonic \
	  tests/operators/mutation/test_real_mutation.py \
	  tests/operators/mutation/test_evosax.py::TestEvosaxAblationIntegrity::test_evosax_wrapper_identity \
	  tests/operators/mutation/test_evosax_mutation_parity.py::test_mutation_matches_evosax_direct

run_failing:
	@echo "--- Rerunning only the tests that failed last time (uses pytest cache) ---"
	python -m pytest --last-failed -v

test-fixes:
	@echo "--- Running recently fixed tests ---"
	python -m pytest --no-cov -v \
	  tests/benchmarks/test_cli.py::TestLoadConfig::test_load_missing_file \
	  tests/operators/mutation/test_real_mutation.py::TestRealMutationHarness::test_jit_reproducibility

lint:
	@echo "--- Checking code quality with Ruff ---"
	ruff check .

format:
	@echo "--- Formatting code with Ruff ---"
	ruff format .

format-check:
	@echo "--- Checking formatting (no mutations) ---"
	ruff format --check .

type-check:
	@echo "--- Running mypy type checker ---"
	mypy src

check-all: lint format-check type-check test
	@echo "--- All checks passed! ---"

bench:
	@echo "--- Running dispatch timing benchmark ---"
	python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --quick

test-bench:
	@echo "--- Running benchmark unit tests (functional, no timing harness) ---"
	python -m pytest tests/benchmarks/ --no-cov -v --tb=short --ignore=tests/benchmarks/test_snapshot_benchmark.py

test-bench-snapshot:
	@echo "--- Running pytest-benchmark snapshot suite (requires: make install-bench) ---"
	python -m pytest tests/benchmarks/test_snapshot_benchmark.py --no-cov -v

docs:
	@echo "--- Building Sphinx HTML documentation ---"
	sphinx-build -b html docs/source docs/build/html

define run_nohup
	@logfile=$$(date +%Y%m%d_%H%M%S)_$1__nohup.log; \
	echo "--- Running $1 (nohup) output -> $$logfile ---"; \
	nohup $(2) > $$logfile 2>&1 &
endef

test-nohup:
	$(call run_nohup,test,python -m pytest)

test-fast-nohup:
	$(call run_nohup,test-fast,python -m pytest --no-cov -q)


test-bench-nohup:
	$(call run_nohup,test-bench,python -m pytest tests/benchmarks/ --no-cov -v --tb=short --ignore=tests/benchmarks/test_snapshot_benchmark.py)

bench-nohup:
	$(call run_nohup,bench,python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --quick)


test-bench-snapshot-nohup:
	$(call run_nohup,test-bench-snapshot,python -m pytest tests/benchmarks/test_snapshot_benchmark.py --no-cov -v)
