# BBOB Benchmark Suite

Comprehensive benchmarking framework for comparing MalthusJAX and Evosax strategies across all 24 BBOB functions.

## Overview

This benchmark suite:
- Tests **24 BBOB functions** across **2 dimensions (10D, 50D)**
- Compares **7 evolutionary strategies** (4 MalthusJAX + 3 Evosax)
- Uses **8 population sizes** (1023, 1024, 1025, 1026, 511, 512, 513, 515)
- Runs **100 independent seeds** per configuration
- Generates and launches **~460,800 evolutionary runs**
- Automatically cleans up RAM after each experiment

## Components

### 1. `generate_bbob_benchmark.py`

Generates TOML experiment files for all BBOB functions.

#### Usage

```bash
# Generate all 24 function TOML files (default settings)
python scripts/generate_bbob_benchmark.py

# Generate with custom settings
python scripts/generate_bbob_benchmark.py \
  --output-dir examples/_DEMO_COMPOSER/bbob_benchmark \
  --fn-range 1 24 \
  --dimensions 10 50 \
  --pop-sizes 1023 1024 1025 1026 511 512 513 515 \
  --num-seeds 100 \
  --generations 500 \
  --create-launcher
```

#### Output

- `bbob_fn01.toml` through `bbob_fn24.toml` — One TOML file per BBOB function
- Each TOML contains ~192 pipelines (2 dims × 8 pop sizes × 12 strategies)
- Launcher script for automated execution (with `--create-launcher`)

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | `examples/_DEMO_COMPOSER/bbob_benchmark` | Output directory for TOML files |
| `--fn-range START END` | `1 24` | BBOB function range (both inclusive) |
| `--dimensions D1 D2 ...` | `10 50` | Dimensions to test |
| `--pop-sizes P1 P2 ...` | `1023 1024 1025 1026 511 512 513 515` | Population sizes |
| `--num-seeds N` | `100` | Seeds per experiment |
| `--generations G` | `500` | Evolutionary generations |
| `--create-launcher` | — | Generate bash launcher script |

### 2. `launch_bbob_benchmark.py`

Launches all experiments with parallel execution and automatic RAM cleanup.

#### Usage

```bash
# Launch with sequential execution
python scripts/launch_bbob_benchmark.py \
  --toml-dir examples/_DEMO_COMPOSER/bbob_benchmark

# Launch with 2 parallel jobs and RAM cleanup
python scripts/launch_bbob_benchmark.py \
  --toml-dir examples/_DEMO_COMPOSER/bbob_benchmark \
  --max-parallel 2 \
  --cleanup-ram

# Launch with custom output directory
python scripts/launch_bbob_benchmark.py \
  --toml-dir examples/_DEMO_COMPOSER/bbob_benchmark \
  --output-dir examples/_DEMO_COMPOSER \
  --max-parallel 4
```

#### Features

- **Parallel execution**: Run multiple experiments in parallel (default: 1)
- **RAM cleanup**: Automatically clean up memory after each run
- **Process tracking**: Monitor PID and execution time per experiment
- **Logging**: Complete logs in `logs/completion.log`
- **Nohup output**: Capture stdout/stderr for each run
- **Memory reporting**: Display memory usage before/after cleanup

#### Output

```
examples/_DEMO_COMPOSER/
├── nohup/
│   ├── bbob_fn01.out
│   ├── bbob_fn02.out
│   └── ... (stdout/stderr for each run)
├── logs/
│   └── completion.log  (summary of all runs)
└── bbob_benchmark/
    ├── fn01/
    ├── fn02/
    └── ...  (results for each function)
```

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--toml-dir` | `examples/_DEMO_COMPOSER/bbob_benchmark` | TOML files directory |
| `--output-dir` | parent of toml-dir | Output/logging directory |
| `--max-parallel N` | `1` | Maximum parallel experiments |
| `--cleanup-ram` | True | Enable RAM cleanup after each run |
| `--no-cleanup-ram` | — | Disable RAM cleanup |

### 3. `bbob_benchmark_workflow.py`

Unified interface combining generation, launching, and analysis.

#### Usage

```bash
# Full workflow: generate + launch
python scripts/bbob_benchmark_workflow.py --generate --launch --max-parallel 2

# Generate only (functions 1-5 for testing)
python scripts/bbob_benchmark_workflow.py --generate --fn-start 1 --fn-end 5

# Launch only (pre-generated files)
python scripts/bbob_benchmark_workflow.py --launch --max-parallel 4

# Analyze results
python scripts/bbob_benchmark_workflow.py --analyze
```

#### Configuration

All parameters from `generate_bbob_benchmark.py` and `launch_bbob_benchmark.py`:

```bash
python scripts/bbob_benchmark_workflow.py \
  --generate \
  --launch \
  --output-dir examples/_DEMO_COMPOSER \
  --fn-start 1 \
  --fn-end 24 \
  --dimensions 10 50 \
  --pop-sizes 1023 1024 1025 1026 511 512 513 515 \
  --num-seeds 100 \
  --generations 500 \
  --max-parallel 2 \
  --cleanup-ram
```

## Strategies Included

### MalthusJAX Strategies (4)

1. **malthusjax_default**
   - Selection: Elite pool (top 50%)
   - Crossover: Uniform
   - Mutation: Gaussian (rate 0.1)

2. **malthusjax_roulette**
   - Selection: Fitness-proportional (roulette wheel)
   - Crossover: Uniform
   - Mutation: Gaussian (rate 0.1)

3. **malthusjax_tournament**
   - Selection: Tournament (size 3)
   - Crossover: Uniform
   - Mutation: Gaussian (rate 0.1)

4. **malthusjax_evosaxops**
   - Selection: Elite pool (top 50%)
   - Crossover: Evosax uniform
   - Mutation: Evosax Gaussian

### Evosax Strategies (3)

1. **evosax_simplega** — Simple Genetic Algorithm
2. **evosax_de** — Differential Evolution
3. **evosax_mr15ga** — MR15 Genetic Algorithm

## TOML Structure

Each generated TOML file has this structure:

```toml
[experiment]
name = "bbob_fn01_sweep"
output_dir = "results/bbob_benchmark/fn01"
description = "BBOB Function 1: sweep across dims=10/50, pop_sizes=511-1026, strategies"

[experiment.shared]
fitness = "bbob:fn=1"          # Will be overridden per dimension
engine_type = "ga"
genome_type = "real"
bounds = [-5.0, 5.0]
elitism = 0
track_best = 0
generations = 500
seeds = [1, 2, ..., 100]

# 192 pipelines: 2 dims × 8 pop_sizes × 12 strategies
[pipelines.malthusjax_default_10d_1024]
genome_length = 10
pop_size = 1024
backend = "malthusjax"
selection = "elite_pool:num_selections=100,elite_k=50"
crossover = "uniform_real"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.evosax_de_50d_1025]
genome_length = 50
pop_size = 1025
backend = "evosax"
evosax_strategy = "DifferentialEvolution"

# ... (190 more pipelines)
```

## Example Workflows

### Quick Test (1 Function, Sequential)

```bash
# Generate function 1 only
python scripts/generate_bbob_benchmark.py --fn-range 1 1

# Launch with 1 parallel job
python scripts/launch_bbob_benchmark.py \
  --toml-dir examples/_DEMO_COMPOSER/bbob_benchmark \
  --max-parallel 1
```

**Estimated time**: ~30 minutes (sequential, 100 seeds)

### Full Benchmark (All 24 Functions, 2 Parallel)

```bash
python scripts/bbob_benchmark_workflow.py \
  --generate \
  --launch \
  --max-parallel 2
```

**Estimated time**: 
- Generation: ~1 minute
- Execution: ~24 hours (with 2 parallel jobs)
- **Total: ~25 hours**

### High-Throughput (4 Parallel, No RAM Cleanup)

```bash
python scripts/bbob_benchmark_workflow.py \
  --launch \
  --max-parallel 4 \
  --no-cleanup-ram
```

**Requirements**: High memory system (8+ GB per experiment)

## Results Structure

```
results/
└── bbob_benchmark/
    ├── fn01/  # BBOB Function 1
    │   ├── traces/        # JAX profiler traces
    │   ├── run_logs/      # Execution logs
    │   └── results.json   # Aggregated results
    ├── fn02/
    └── ...
```

Each function directory contains:
- **Per-strategy convergence curves** (best fitness vs generation)
- **Aggregated statistics** across seeds (mean, median, std)
- **Population size sensitivity** analysis
- **Strategy comparison** plots

## Performance Considerations

### Memory Usage

- **Per experiment** (1 seed): ~500MB-2GB (depending on pop_size)
- **Parallel factor**: 500MB-2GB per parallel job
- **Recommendation**: 1 parallel job per 4GB available RAM

### Disk Space

- **Per function** (100 seeds): ~200MB
- **All 24 functions**: ~5GB
- **With traces**: +20GB

### Time Estimates

| Configuration | Time | Notes |
|---------------|------|-------|
| 1 function, sequential | 30 min | Baseline |
| 24 functions, sequential | 12 hours | Not practical |
| 24 functions, 2 parallel | ~6 hours | Recommended |
| 24 functions, 4 parallel | ~3 hours | Requires high memory |

## Troubleshooting

### Issue: "No space left on device"

**Solution**: Increase available disk space or reduce number of seeds:

```bash
python scripts/generate_bbob_benchmark.py --num-seeds 30
```

### Issue: "Memory error" during execution

**Solution**: Reduce parallelism or enable RAM cleanup:

```bash
python scripts/launch_bbob_benchmark.py \
  --toml-dir bbob_benchmark \
  --max-parallel 1 \
  --cleanup-ram
```

### Issue: Experiments not launching

**Solution**: Check that TOML files were generated:

```bash
ls -la examples/_DEMO_COMPOSER/bbob_benchmark/
```

If empty, run generator first:

```bash
python scripts/generate_bbob_benchmark.py
```

### Issue: Viewing progress

**Solution**: Check completion log in real-time:

```bash
tail -f examples/_DEMO_COMPOSER/logs/completion.log
```

## Advanced: Custom Configurations

### Test Specific Functions

```bash
python scripts/generate_bbob_benchmark.py \
  --fn-range 1 5 \
  --num-seeds 30  # Faster testing
```

### Different Population Sizes

```bash
python scripts/generate_bbob_benchmark.py \
  --pop-sizes 256 512 1024 2048 4096
```

### Custom Dimensions

```bash
python scripts/generate_bbob_benchmark.py \
  --dimensions 2 5 10 20
```

## Citation

If you use this benchmark suite, cite:

```bibtex
@software{malthusjax_bbob,
  title={BBOB Benchmark Suite for MalthusJAX},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/MalthusJAX}
}
```

## See Also

- [BBOB Documentation](https://numbbo.github.io/bbob-doc/)
- [Evosax Documentation](https://github.com/RobertTLange/evosax)
- [MalthusJAX TOML Grammar Guide](../docs/toml_grammar_guide.md)
