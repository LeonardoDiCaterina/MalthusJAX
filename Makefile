.PHONY: help install-dev install-bench test test-fast test-failing test-fixes test-bench test-bench-snapshot \
        test-bench-group-01 test-bench-group-02 test-bench-group-03 test-bench-group-04 \
        test-bench-group-05 test-bench-group-06 test-bench-group-07 test-bench-group-08 \
        test-bench-group-09 test-bench-group-10 test-bench-group-11 \
        lint format format-check type-check check-all docs docs-clean docs-open bench

# --- Auto-Detect CUDA Version ---
HAS_NVIDIA := $(shell command -v nvidia-smi 2> /dev/null)
ifdef HAS_NVIDIA
    # Extracts the major version number (e.g., 11, 12, 13)
    CUDA_MAJOR := $(shell nvidia-smi | grep -Eo 'CUDA Version: [0-9]+' | awk '{print $$3}')
    
    ifeq ($(CUDA_MAJOR),13)
        JAX_EXTRA := cuda13
    else ifeq ($(CUDA_MAJOR),12)
        JAX_EXTRA := cuda12
    else ifeq ($(CUDA_MAJOR),11)
        JAX_EXTRA := cuda11
    else
        JAX_EXTRA := cpu
    endif
else
    JAX_EXTRA := cpu
endif
# --------------------------------

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
	@echo ""
	@echo "--- Documentation Workflow ---"
	@echo "  make docs                    Build Sphinx HTML docs (picks up all changes)"
	@echo "  make docs-clean              Remove docs/build/ (clean rebuild)"
	@echo "  make docs-open               Open built docs in browser"
	@echo "  make -C docs apidoc-force    Regenerate API stubs after API changes"
	@echo ""
	@echo "Documentation Update Guide:"
	@echo "  • Changed Python API?        → make -C docs apidoc-force && make docs"
	@echo "  • Edited markdown files?     → make docs (rebuilds automatically)"
	@echo "  • Updated docstrings?        → make docs (autodoc picks them up)"
	@echo "  • Full clean rebuild?        → make docs-clean && make docs"
	@echo ""
	@echo "Markdown docs live in: docs/source/*.md (synced from docs/)"
	@echo "API stubs auto-generated in:  docs/source/api/ (don't edit manually)"
	@echo "Built site output in:         docs/build/html/"

PYTHON ?= python

install-dev:
	@echo "--- Installing dev dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[dev,docs,examples,$(JAX_EXTRA)]"

install-bench:
	@echo "--- Installing benchmark dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[dev,benchmarks,$(JAX_EXTRA)]"

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
	python -m pytest tests/benchmarks/test_benchmark_*.py --no-cov -v

# individual group targets for convenience

test-bench-group-01:
	@echo "--- Running single-step latency benchmarks ---"
	pytest tests/benchmarks/test_benchmark_01_single_step.py --no-cov -v --benchmark-only

test-bench-group-02:
	@echo "--- Running multi-gen throughput benchmarks ---"
	pytest tests/benchmarks/test_benchmark_02_multi_gen_throughput.py --no-cov -v --benchmark-only

test-bench-group-03:
	@echo "--- Running JIT compilation benchmarks ---"
	pytest tests/benchmarks/test_benchmark_03_compilation.py --no-cov -v --benchmark-only

test-bench-group-04:
	@echo "--- Running operator microbenchmark tests ---"
	pytest tests/benchmarks/test_benchmark_04_operators.py --no-cov -v --benchmark-only

test-bench-group-05:
	@echo "--- Running convergence parity benchmarks ---"
	pytest tests/benchmarks/test_benchmark_05_convergence.py --no-cov -v --benchmark-only

test-bench-group-06:
	@echo "--- Running unroll factor sweep benchmarks ---"
	pytest tests/benchmarks/test_benchmark_06_unroll.py --no-cov -v --benchmark-only

test-bench-group-07:
	@echo "--- Running phase breakdown benchmarks ---"
	pytest tests/benchmarks/test_benchmark_07_phases.py --no-cov -v --benchmark-only

test-bench-group-08:
	@echo "--- Running scaling sweep benchmarks ---"
	pytest tests/benchmarks/test_benchmark_08_scaling.py --no-cov -v --benchmark-only

test-bench-group-09:
	@echo "--- Running injection operator performance benchmarks ---"
	pytest tests/benchmarks/test_benchmark_09_injection.py --no-cov -v --benchmark-only

test-bench-group-10:
	@echo "--- Running key derivation strategy benchmarks ---"
	pytest tests/benchmarks/test_benchmark_10_key_derivation.py --no-cov -v --benchmark-only

test-bench-group-11:
	@echo "--- Running injection + key derivation parity benchmarks ---"
	pytest tests/benchmarks/test_benchmark_11_injection_parity.py --no-cov -v --benchmark-only

# nohup variants for each group

test-bench-group-01-nohup:
	$(call run_nohup,test-bench-group-01,make test-bench-group-01)

test-bench-group-02-nohup:
	$(call run_nohup,test-bench-group-02,make test-bench-group-02)

test-bench-group-03-nohup:
	$(call run_nohup,test-bench-group-03,make test-bench-group-03)

test-bench-group-04-nohup:
	$(call run_nohup,test-bench-group-04,make test-bench-group-04)

test-bench-group-05-nohup:
	$(call run_nohup,test-bench-group-05,make test-bench-group-05)

test-bench-group-06-nohup:
	$(call run_nohup,test-bench-group-06,make test-bench-group-06)

test-bench-group-07-nohup:
	$(call run_nohup,test-bench-group-07,make test-bench-group-07)

test-bench-group-08-nohup:
	$(call run_nohup,test-bench-group-08,make test-bench-group-08)

test-bench-group-09-nohup:
	$(call run_nohup,test-bench-group-09,make test-bench-group-09)

test-bench-group-10-nohup:
	$(call run_nohup,test-bench-group-10,make test-bench-group-10)

test-bench-group-11-nohup:
	$(call run_nohup,test-bench-group-11,make test-bench-group-11)

# ============================================================================= #
# Documentation targets
# ============================================================================= #
# Workflow:
#   1. Edit markdown files in docs/source/*.md or docstrings in src/
#   2. If you changed the API (added/removed/renamed modules):
#      → make -C docs apidoc-force (regenerate API stubs)
#   3. Run: make docs (rebuilds the entire site with verbose output)
#   4. View: open docs/build/html/index.html
#
# See also: docs/Makefile for sphinx-specific targets (apidoc, apidoc-force, etc.)
# ============================================================================= #

docs:
	@echo "--- Building Sphinx HTML documentation ---"
	python -m sphinx -v -b html docs/source docs/build/html

docs-clean:
	@echo "--- Cleaning docs build directory ---"
	rm -rf docs/build

docs-open:
	@echo "--- Opening docs in browser ---"
	open docs/build/html/index.html

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
