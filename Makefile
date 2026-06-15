.PHONY: help install-dev install-bench test test-fast test-failing test-fixes test-bench test-bench-snapshot \
        test-bench-group-01 test-bench-group-02 test-bench-group-03 test-bench-group-04 \
        test-bench-group-05 test-bench-group-06 test-bench-group-07 test-bench-group-08 \
        test-bench-group-09 test-bench-group-10 test-bench-group-11 \
	lint format format-check type-check check-all docs docs-clean docs-open \
	run-toml run-toml-nohup run-toml-with-artifacts list-toml \
	parity-toml \
	artifacts-toml artifacts-dir artifacts-batch \
	toy-100seeds toy-100seeds-sphere-d5

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
	@echo "  make *-nohup           Run the same command under nohup and log output to a timestamped file"
	@echo "  make lint               Ruff lint check (no fixes)"
	@echo "  make format             Ruff auto-format (mutates files)"
	@echo "  make format-check       Ruff format check only (no mutations)"
	@echo "  make type-check         mypy strict check on src/"
	@echo "  make check-all          lint + format-check + type-check + test"
	@echo ""
	@echo "--- Experiment Execution (TOML-based) ---"
	@echo "  make run-toml TOML=<file>           Run experiment from TOML file"
	@echo "  make run-toml-with-artifacts TOML=<file>  Run TOML and auto-generate artifacts"
	@echo "  make run-toml-nohup TOML=<file>     Run TOML experiment in background (logs to .log)"
	@echo "  make parity-toml TOML=<file>        Run two-pipeline statistical parity from TOML"
	@echo "    + PLOT=1 to generate default parity plots"
	@echo "    + PLOT_EXTRA=1 to add delta + Bland-Altman diagnostics"
	@echo "  make suite-parity CONFIG_DIR=<dir> OUT_DIR=<dir>  Run parity on a folder of TOMLs and aggregate"
	@echo "  make artifacts-toml TOML=<file>     Generate artifacts for TOML output_dir"
	@echo "  make artifacts-dir RESULTS_DIR=<dir>  Generate artifacts for one results dir"
	@echo "  make artifacts-batch RESULTS_GLOB=<glob> Generate artifacts for matching dirs"
	@echo "  make list-toml                      List available TOML experiment files"
	@echo "  make toy-100seeds                  Run legacy 100-seed toy matrix (d=3)"
	@echo "  make toy-100seeds-sphere-d5        Run legacy 100-seed toy matrix (sphere d=5)"
	@echo ""
	@echo "--- Artifact Generation Details ---"
	@echo "  Artifacts are written to: <results_dir>/artifacts/"
	@echo "  For multi-pipeline runs, comparison artifacts are generated:" 
	@echo "    - comparison_table.csv / comparison_table.md"
	@echo "    - comparison_final_best_fitness_boxplot.png"
	@echo "    - comparison_timing_boxplot.png"
	@echo "    - comparison_convergence_best_fitness.png"
	@echo "  Example batch command:"
	@echo "    make artifacts-batch RESULTS_GLOB='results/bbob_*_pop1024'"
	@echo ""
	@echo "Example:"
	@echo "  make run-toml TOML=configs/examples/sphere_experiment.toml"
	@echo "  make suite-parity CONFIG_DIR=configs/thesis/ OUT_DIR=results/thesis_suite"
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
TOY_SEED_START ?= 0
TOY_SEED_END ?= 99
TOY_POP_SIZE ?= 12
TOY_GENERATIONS ?= 20

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

test-bench-snapshot-nohup:
	$(call run_nohup,test-bench-snapshot,python -m pytest tests/benchmarks/test_snapshot_benchmark.py --no-cov -v)

# ============================================================================= #
# TOML-Based Experiment Execution
# ============================================================================= #

run-toml:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-toml TOML=configs/examples/experiment.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	@echo "--- Running TOML experiment: $(TOML) ---"
	mjax run $(TOML)

run-toml-nohup:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-toml-nohup TOML=configs/examples/experiment.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	$(call run_nohup,toml_experiment,mjax run $(TOML))

parity-toml:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make parity-toml TOML=configs/parity/parity.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	@echo "--- Running statistical parity for $(TOML) ---"
	mjax parity $(TOML)

artifacts-dir:
	@test -n "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR variable not set"; echo "Usage: make artifacts-dir RESULTS_DIR=results/my_experiment"; exit 1)
	@test -d "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR not found: $(RESULTS_DIR)"; exit 1)
	@echo "--- Generating artifacts for $(RESULTS_DIR) ---"
	mjax report $(RESULTS_DIR)

list-toml:
	@echo "--- Available TOML experiment files ---"
	@find configs -type f -name "*.toml" | sort | while read f; do \
		echo "  $$f"; \
	done

suite-parity:
	@test -n "$(CONFIG_DIR)" || (echo "Error: CONFIG_DIR variable not set"; echo "Usage: make suite-parity CONFIG_DIR=configs/thesis/ OUT_DIR=results/my_suite"; exit 1)
	@test -n "$(OUT_DIR)" || (echo "Error: OUT_DIR variable not set"; echo "Usage: make suite-parity CONFIG_DIR=configs/thesis/ OUT_DIR=results/my_suite"; exit 1)
	@test -d "$(CONFIG_DIR)" || (echo "Error: CONFIG_DIR not found: $(CONFIG_DIR)"; exit 1)
	@echo "--- Running statistical parity suite from $(CONFIG_DIR) ---"
	@for toml in $(CONFIG_DIR)/*.toml; do \
		if [ -f "$$toml" ]; then \
			$(PYTHON) -m malthusjax.benchmarking.cli parity $$toml; \
		fi \
	done
	@echo "--- Aggregating results into $(OUT_DIR) ---"
	@DIRS=""; \
	for toml in $(CONFIG_DIR)/*.toml; do \
		if [ -f "$$toml" ]; then \
			name=$$(basename $$toml .toml); \
			DIRS="$$DIRS results/$$name"; \
		fi \
	done; \
	$(PYTHON) -m malthusjax.benchmarking.cli aggregate --out_dir $(OUT_DIR) $$DIRS
	@echo "--- Generating LaTeX table ---"
	@$(PYTHON) scripts/generate_parity_latex.py --suite_dir $(OUT_DIR) --out $(OUT_DIR)/parity_table.tex

# Include experiment helper targets (kept modular for clarity)
INCLUDE_FROM_ROOT := 1
include Makefile.experiments

# ============================================================================= #
# Thesis Experiments Execution
# ============================================================================= #
thesis-experiments:
	@echo "--- Running Full Thesis Experiment Suite ---"
	$(PYTHON) scripts/run_thesis_experiments.py
	@echo "--- Thesis experiments completed successfully. ---"

thesis-experiments-nohup:
	$(call run_nohup,thesis-experiments,make thesis-experiments)

thesis-experiments-smoke:
	@echo "--- Running SMOKE TEST Thesis Experiment Suite ---"
	SMOKE_TEST=1 $(PYTHON) scripts/run_thesis_experiments.py
	@echo "--- Smoke test completed successfully. ---"

# ============================================================================= #
# Thesis LHS & Diagnostics Master Pipeline
# ============================================================================= #

thesis-lhs-configs:
	@echo "--- Generating LHS configurations ---"
	$(PYTHON) scripts/thesis_reproducibility/generate_lhs_configs.py

thesis-lhs-run:
	@echo "--- Running all LHS thesis experiments ---"
	@for toml in configs/lhs_experiments/*.toml; do \
		if [ -f "$$toml" ]; then \
			$(PYTHON) -m malthusjax.benchmarking.cli run "$$toml"; \
		fi \
	done

thesis-lhs-regression:
	@echo "--- Running LHS regression diagnostics ---"
	$(PYTHON) scripts/thesis_reproducibility/analyze_lhs_regression.py

thesis-master-run: thesis-lhs-configs thesis-lhs-run thesis-lhs-regression thesis-chapter-compile


thesis-chapter-compile:
	@echo "--- Compiling final Markdown chapter ---"
	$(PYTHON) scripts/thesis_reproducibility/compile_markdown_chapter.py

