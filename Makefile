.PHONY: help install-dev install-bench test test-fast test-unit test-failing test-fixes test-bench test-bench-snapshot \
	lint format format-check type-check check-all docs docs-clean docs-open \
	demo reproduce reproduce-nohup smoke-all \
	perf-bench perf-bench-smoke perf-hlo perf-perfetto perf-tb perf-tb-bg perf-all perf-all-nohup

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
	@echo "  make test-unit      Run fast unit tests only (skips slow tests)"
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
	@echo ""
	@echo "--- Thesis & Show & Tell ---"
	@echo "  make show-tell                      Run the full thesis parity evaluation (H1, H2, H3)"
	@echo "  make show-tell-smoke                Run the smoke test version of the thesis evaluation"
	@echo "  make smoke-all                      Run smoke tests for all unified TOML benchmarks"
	@echo ""
	@echo "--- Docker Execution (Cluster) ---"
	@echo "  make docker-build                   Build the GPU-enabled Docker image"
	@echo "  make docker-h<1|2|3>-<...>-full     Run a specific thesis suite in a detached container"
	@echo "  make docker-all-full                Run ALL thesis suites sequentially in a detached container"
	@echo "  make docker-all-smoke               Run ALL smoke tests sequentially in a detached container"
	@echo "  (All docker targets automatically mount results/ and logs/ and print the tail command)"
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
	@echo "  make docker-all-full"
	@echo ""
	@echo "--- Performance Harness (MJX vs EvoSAX) ---"
	@echo "  make perf-bench         Run quality+timing benchmark (TOML pipelines vs EvoSAX)"
	@echo "  make perf-bench-smoke   Smoke version: 3 seeds, tiny problem, fast"
	@echo "  make perf-hlo           Extract & compare XLA HLO graphs for all pipelines"
	@echo "  make perf-perfetto      Generate Perfetto traces (one per pipeline, isolated)"
	@echo "  make perf-tb            Launch TensorBoard to view Perfetto traces (blocking)"
	@echo "  make perf-tb-bg         Same, but launch TensorBoard in background"
	@echo "  make perf-all           Run bench → hlo → perfetto sequentially"
	@echo "  make perf-all-nohup     Same, but headless (nohup)"
	@echo ""
	@echo "  Key variables (override on command line):"
	@echo "    PERF_TOML  (default: $(PERF_TOML))"
	@echo "    PERF_OUT   (default: <suite.output_dir from TOML>)"
	@echo "    PERF_DIMS  (default: $(PERF_DIMS))"
	@echo "    PERF_POP   (default: $(PERF_POP))"
	@echo "    PERF_GENS  (default: $(PERF_GENS))"
	@echo "    PORT       (default: $(PORT))"
	@echo ""
	@echo "  Workflow:"
	@echo "    1. Add a new engine variant to configs/perf/h1_speed_vs_evosax.toml"
	@echo "    2. make perf-hlo      → check XLA graph complexity vs EvoSAX baseline"
	@echo "    3. make perf-bench    → verify solution quality parity"
	@echo "    4. make perf-perfetto → profile kernel execution"
	@echo "    5. make perf-tb PORT=<port> → inspect in TensorBoard"
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
MJAX_CMD ?= PYTHONPATH=src $(PYTHON) -m malthusjax.benchmarking.cli
# ==============================================================================
# PERFORMANCE HARNESS — MJX vs EvoSAX speed optimization loop
# All variables can be overridden on the command line:
#   make perf-bench PERF_TOML=configs/perf/h1_speed_vs_evosax.toml PORT=6007
# ==============================================================================
PERF_TOML  ?= configs/perf/h1_speed_vs_evosax.toml
PERF_SMOKE ?= configs/perf/smoke_speed_vs_evosax.toml
PERF_OUT   ?= results/perf/$(shell basename $(PERF_TOML) .toml)
PERF_DIMS  ?= 9
PERF_POP   ?= 195
PERF_GENS  ?= 387
PORT       ?= 6006

TOY_SEED_START ?= 0
TOY_SEED_END ?= 99
TOY_POP_SIZE ?= 12
TOY_GENERATIONS ?= 20

install-dev:
	@echo "--- Installing dev dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[dev,docs,examples,all-integrations,$(JAX_EXTRA)]"

install-bench:
	@echo "--- Installing benchmark dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[dev,benchmarks,all-integrations,$(JAX_EXTRA)]"

install-stats:
	@echo "--- Installing statistical dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[stats,$(JAX_EXTRA)]"

install-rl:
	@echo "--- Installing RL dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[rl,$(JAX_EXTRA)]"

install-evosax:
	@echo "--- Installing evosax dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[evosax,$(JAX_EXTRA)]"

install-qdax:
	@echo "--- Installing qdax dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[qdax,$(JAX_EXTRA)]"

install-tensorneat:
	@echo "--- Installing tensorneat dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[tensorneat,$(JAX_EXTRA)]"

install-kozax:
	@echo "--- Installing kozax dependencies (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[kozax,$(JAX_EXTRA)]"

install-all:
	@echo "--- Installing all integrations (Detected Backend: $(JAX_EXTRA)) ---"
	$(PYTHON) -m pip install -e ".[all-integrations,$(JAX_EXTRA)]"

test:
	@echo "--- Running tests with coverage ---"
	python -m pytest

test-fast:
	@echo "--- Running tests without coverage ---"
	python -m pytest --no-cov -q

test-unit:
	@echo "--- Running fast unit tests only ---"
	python -m pytest -m "not slow" --no-cov -q

test-failing:
	@echo "--- Running previously failing test subset ---"
	python -m pytest --no-cov -q \
	  tests/composer/test_engine_factory.py::test_engine_adapter_run_once \
	  tests/composer/test_optimization_direction.py::TestOptimizationDirection::test_optimization_directions_are_opposite \
	  tests/engine/test_genetic_engine.py \
	  tests/engine/test_genetic_engine_asktel.py \
	  tests/engine/test_genetic_engine_core.py \
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
	$(call bg_task,test-bench-group-01,make test-bench-group-01)

test-bench-group-02-nohup:
	$(call bg_task,test-bench-group-02,make test-bench-group-02)

test-bench-group-03-nohup:
	$(call bg_task,test-bench-group-03,make test-bench-group-03)

test-bench-group-04-nohup:
	$(call bg_task,test-bench-group-04,make test-bench-group-04)

test-bench-group-05-nohup:
	$(call bg_task,test-bench-group-05,make test-bench-group-05)

test-bench-group-06-nohup:
	$(call bg_task,test-bench-group-06,make test-bench-group-06)

test-bench-group-07-nohup:
	$(call bg_task,test-bench-group-07,make test-bench-group-07)

test-bench-group-08-nohup:
	$(call bg_task,test-bench-group-08,make test-bench-group-08)

test-bench-group-09-nohup:
	$(call bg_task,test-bench-group-09,make test-bench-group-09)

test-bench-group-10-nohup:
	$(call bg_task,test-bench-group-10,make test-bench-group-10)

test-bench-group-11-nohup:
	$(call bg_task,test-bench-group-11,make test-bench-group-11)

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

define bg_task
	@mkdir -p results
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	LOGFILE="results/$$1_$$TIMESTAMP.log"; \
	nohup $$2 > $$LOGFILE 2>&1 & \
	PID=$$!; \
	echo "
======================================================="; \
	echo " BACKGROUND TASK STARTED: $$1"; \
	echo "======================================================="; \
	echo " PID: $$PID"; \
	echo " Log file: $$LOGFILE"; \
	echo ""; \
	echo " To watch the logs live, run:"; \
	echo "    tail -f $$LOGFILE"; \
	echo ""; \
	echo " To kill the process, run:"; \
	echo "    kill $$PID"; \
	echo "=======================================================
"
endef

test-nohup:
	$(call bg_task,test,python -m pytest)

test-fast-nohup:
	$(call bg_task,test-fast,python -m pytest --no-cov -q)


test-bench-nohup:
	$(call bg_task,test-bench,python -m pytest tests/benchmarks/ --no-cov -v --tb=short --ignore=tests/benchmarks/test_snapshot_benchmark.py)

test-bench-snapshot-nohup:
	$(call bg_task,test-bench-snapshot,python -m pytest tests/benchmarks/test_snapshot_benchmark.py --no-cov -v)

# ============================================================================= #
# TOML-Based Experiment Execution
# ============================================================================= #

run-toml:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-toml TOML=configs/examples/experiment.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	@echo "--- Running TOML experiment: $(TOML) ---"
	$(MJAX_CMD) run $(TOML)

run-toml-nohup:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-toml-nohup TOML=configs/examples/experiment.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	$(call bg_task,toml_experiment,$(MJAX_CMD) run $(TOML))

run-benchmark:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-benchmark TOML=configs/h1_parity_cartesian.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	@echo "--- Running Benchmark Suite: $(TOML) ---"
	python scripts/benchmark_runner.py --toml $(TOML)

run-benchmark-nohup:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make run-benchmark-nohup TOML=configs/h1_parity_cartesian.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	$(call bg_task,benchmark_suite,python scripts/benchmark_runner.py --toml $(TOML))

parity-toml:
	@test -n "$(TOML)" || (echo "Error: TOML variable not set"; echo "Usage: make parity-toml TOML=configs/parity/parity.toml"; exit 1)
	@test -f "$(TOML)" || (echo "Error: TOML file not found: $(TOML)"; exit 1)
	@echo "--- Running statistical parity for $(TOML) ---"
	$(MJAX_CMD) parity $(TOML)

artifacts-dir:
	@test -n "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR variable not set"; echo "Usage: make artifacts-dir RESULTS_DIR=results/my_experiment"; exit 1)
	@test -d "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR not found: $(RESULTS_DIR)"; exit 1)
	@echo "--- Generating artifacts for $(RESULTS_DIR) ---"
	$(MJAX_CMD) report $(RESULTS_DIR)

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

thesis-tables:
	@test -n "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR variable not set"; echo "Usage: make thesis-tables RESULTS_DIR=results/h1_parity_qdax"; exit 1)
	@test -d "$(RESULTS_DIR)" || (echo "Error: RESULTS_DIR not found: $(RESULTS_DIR)"; exit 1)
	@echo "--- Generating Thesis Tables for $(RESULTS_DIR) ---"
	@$(PYTHON) scripts/generate_thesis_tables.py --dir $(RESULTS_DIR)

# ============================================================================= #
# Docker Execution (Cluster)
# ============================================================================= #

docker-build:
	@echo "--- Building Docker Image with GPU Support ---"
	docker build --build-arg EXTRAS="[cuda12,qdax]" -t malthusjax-gpu .

docker-reproduce:
	@echo "--- Running Reproduce via Docker (Detached) ---"
	@docker rm -f docker-reproduce 2>/dev/null || true
	docker run -d --gpus all --name docker-reproduce --entrypoint make -v $$(pwd)/results:/app/results -v $$(pwd)/logs:/app/logs malthusjax-gpu reproduce
	@echo "\n>>> SUCCESS: Container started in background!"
	@echo ">>> To monitor progress live, run:  docker logs -f docker-reproduce\n"

docker-smoke-all:
	@echo "--- Running Smoke All via Docker (Detached) ---"
	@docker rm -f docker-smoke-all 2>/dev/null || true
	docker run -d --gpus all --name docker-smoke-all --entrypoint make -v $$(pwd)/results:/app/results -v $$(pwd)/logs:/app/logs malthusjax-gpu smoke-all
	@echo "\n>>> SUCCESS: Container started in background!"
	@echo ">>> To monitor progress live, run:  docker logs -f docker-smoke-all\n"

# ==============================================================================
# UNIFIED TOML BENCHMARKING ENGINE
# ==============================================================================

# Example: make benchmark-run TOML=configs/h1_parity_lhs.toml
trace-pipeline:
	@if [ -z "$(TOML)" ]; then echo "ERROR: Must provide TOML file (e.g., make trace-pipeline TOML=configs/h2_ablation_lhs.toml)"; exit 1; fi
	@echo "--- PROFILING PIPELINES FROM TOML: $(TOML) ---"
	$(PYTHON) scripts/trace_pipelines.py --toml $(TOML)

benchmark-run:
	@if [ -z "$(TOML)" ]; then echo "ERROR: Must provide TOML file (e.g., make benchmark-run TOML=configs/h1_parity_lhs.toml)"; exit 1; fi
	@echo "--- RUNNING BENCHMARK: $(TOML) ---"
	$(PYTHON) scripts/benchmark_runner.py --toml $(TOML)

benchmark-run-smoke:
	@if [ -z "$(TOML)" ]; then echo "ERROR: Must provide TOML file"; exit 1; fi
	$(PYTHON) scripts/benchmark_runner.py --toml $(TOML) --smoke

benchmark-run-nohup:
	@if [ -z "$(TOML)" ]; then echo "ERROR: Must provide TOML file"; exit 1; fi
	$(call bg_task,benchmark_$(shell basename $(TOML) .toml),$(PYTHON) scripts/benchmark_runner.py --toml $(TOML))

# Automatically run smoke tests and analysis for all three hypotheses locally
# ------------------------------------------------------------------------------
# 1. SHOW & TELL DEMO (Fast Narrative Execution)
# ------------------------------------------------------------------------------
demo:
	@echo "=== 1. MalthusJAX Catalog ==="
	$(MJAX_CMD) catalog
	@echo "\n=== 2. Declarative Execution (Sphere 25D) ==="
	$(MJAX_CMD) run configs/examples/sphere_experiment.toml
	@echo "\n=== 3. Zero-Overhead Architecture Ablation ==="
	$(PYTHON) benchmarks/architecture_ablation.py --smoke
	@echo "\n=== 4. Quality-Diversity (MAP-Elites) Parity ==="
	$(PYTHON) benchmarks/qd_architecture_ablation.py --smoke

# ------------------------------------------------------------------------------
# 2. PUBLICATION REPRODUCIBILITY (Thesis Benchmarks)
# ------------------------------------------------------------------------------
reproduce:
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h1_parity_lhs.toml
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h2_ablation_final_lhs.toml
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h3_precision_lhs.toml

reproduce-nohup:
	$(call bg_task,reproduce,make reproduce)

smoke-all:
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h1_parity_lhs.toml --smoke
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h2_ablation_final_lhs.toml --smoke
	$(PYTHON) scripts/benchmark_runner.py --toml configs/h3_precision_lhs.toml --smoke

# ==============================================================================
# PERFORMANCE HARNESS
# ==============================================================================

# Run quality + timing benchmark for all perf TOML pipelines.
# Uses benchmark_runner.py (runs in-process, no subprocess isolation needed;
# run_once() already separates compile-warmup from timed execution).
perf-bench:
	@echo "--- PERF BENCH: $(PERF_TOML) ---"
	@echo "  Output → $(PERF_OUT)/bench"
	$(PYTHON) scripts/benchmark_runner.py --toml $(PERF_TOML)
	@echo "\n>>> Done. Results written to: $(PERF_OUT)/bench"

perf-bench-smoke:
	@echo "--- PERF BENCH (smoke): $(PERF_SMOKE) ---"
	$(PYTHON) scripts/benchmark_runner.py --toml $(PERF_SMOKE) --smoke
	@echo "\n>>> Done. Smoke results written to: results/perf/smoke_speed_vs_evosax"

# Extract and compare optimised XLA HLO for all pipelines.
# EvoSAX: JITs strategy.ask()+tell() natively (no adapter overhead).
# MJX:    Calls engine.get_hlo_text() on the compiled scan kernel.
perf-hlo:
	@echo "--- PERF HLO: dims=$(PERF_DIMS)  pop=$(PERF_POP)  gens=$(PERF_GENS) ---"
	@echo "  TOML  → $(PERF_TOML)"
	@echo "  Output → $(PERF_OUT)/hlo"
	$(PYTHON) scripts/extract_hlo.py \
		--toml $(PERF_TOML) \
		--dims $(PERF_DIMS) \
		--pop  $(PERF_POP) \
		--gens $(PERF_GENS) \
		--out-dir $(PERF_OUT)/hlo
	@echo "\n>>> HLO summary written to: $(PERF_OUT)/hlo/hlo_summary.md"

# Generate Perfetto traces for all pipelines (one subprocess per pipeline).
perf-perfetto:
	@echo "--- PERF PERFETTO: $(PERF_TOML) ---"
	@echo "  Output → $(PERF_OUT)/perfetto  (port hint: $(PORT))"
	$(PYTHON) scripts/trace_pipelines.py \
		--toml $(PERF_TOML) \
		--out-dir $(PERF_OUT)/perfetto \
		--port $(PORT)
	@echo "\n>>> Traces written to: $(PERF_OUT)/perfetto"
	@echo ">>> Launch TensorBoard: make perf-tb PORT=$(PORT)"

# Launch TensorBoard pointing at the Perfetto traces (blocking).
perf-tb:
	@echo "--- TensorBoard → $(PERF_OUT)/perfetto  at port $(PORT) ---"
	@echo "  URL: http://localhost:$(PORT)"
	tensorboard --logdir $(PERF_OUT)/perfetto --port $(PORT)

# Same but runs in the background (prints PID).
perf-tb-bg:
	@echo "--- TensorBoard (background) → $(PERF_OUT)/perfetto  at port $(PORT) ---"
	nohup tensorboard --logdir $(PERF_OUT)/perfetto --port $(PORT) \
		> logs/tensorboard_$(PORT).log 2>&1 & \
		echo "TensorBoard PID=$$!  URL=http://localhost:$(PORT)"

# Run the full harness chain sequentially (no TensorBoard).
perf-all:
	@echo "\n=== PERF HARNESS: $(PERF_TOML) ==="
	$(MAKE) perf-bench  PERF_TOML=$(PERF_TOML) PORT=$(PORT)
	$(MAKE) perf-hlo    PERF_TOML=$(PERF_TOML) PERF_DIMS=$(PERF_DIMS) PERF_POP=$(PERF_POP) PERF_GENS=$(PERF_GENS)
	$(MAKE) perf-perfetto PERF_TOML=$(PERF_TOML) PORT=$(PORT)
	@echo "\n=== DONE. Launch TensorBoard: make perf-tb PORT=$(PORT) ==="

perf-all-nohup:
	$(call bg_task,perf-all,make perf-all PERF_TOML=$(PERF_TOML) PORT=$(PORT))

