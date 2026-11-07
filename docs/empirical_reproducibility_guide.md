# MalthusJAX Autonomous Empirical Data Generation & Reproducibility Guide

This document provides the definitive, self-contained guide for reproducing, executing, and benchmarking the empirical results presented in the **MalthusJAX** academic paper (*"MalthusJAX: A High-Throughput, Modular Framework for Accelerator-Native Evolutionary Computation"*).

It covers autonomous execution on GPU clusters (Slurm), local workstation execution, the complete test suite, data synchronization pipelines, and critical systems-level configuration to prevent XLA from exhausting GPU VRAM and host RAM.

---

## Table of Contents
1. [Overview of Empirical Showcases](#1-overview-of-empirical-showcases)
2. [Hardware & Environment Requirements](#2-hardware--environment-requirements)
3. [Autonomous Data Generation Commands](#3-autonomous-data-generation-commands)
   - [Slurm Cluster Batch Execution](#slurm-cluster-batch-execution)
   - [Direct Python Execution](#direct-python-execution)
   - [Execution Modes & Tiers](#execution-modes--tiers)
4. [XLA, JAX & Hardware Memory Optimization](#4-xla-jax--hardware-memory-optimization)
   - [GPU VRAM: Taming Greedy Allocation](#gpu-vram-taming-greedy-allocation)
   - [Host RAM: Preventing Thread Pool Explosions](#host-ram-preventing-thread-pool-explosions)
   - [XLA Compilation Storms & Caching](#xla-compilation-storms--caching)
   - [Asynchronous Dispatch & Memory Accumulation](#asynchronous-dispatch--memory-accumulation)
   - [Headless MuJoCo GPU Offscreen Rendering](#headless-mujoco-gpu-offscreen-rendering)
5. [Complete Verification & Test Suite Commands](#5-complete-verification--test-suite-commands)
6. [Data Synchronization & Paper Compilation Pipeline](#6-data-synchronization--paper-compilation-pipeline)
7. [Troubleshooting & Diagnostic FAQ](#7-troubleshooting--diagnostic-faq)

---

## 1. Overview of Empirical Showcases

The empirical suite exercises five core architectural pillars, writing aggregated metrics to `results/empirical_curves.json` and generating publication-ready LaTeX tables in `results/tables/`:

| Showcase | Benchmark Description | Key Metric / Verification | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Dataset 1A** | **Continuous Control Triad & Dual Parity**<br>MuJoCo MJX Cartpole ($D=4$) & Hopper ($D=5,123$ weights). Tests MalthusJAX GA vs. EvoSAX Open-ES and QDax MAP-Elites, including early-termination cliff ablation. | Exact bitwise floating-point parity with upstream EvoSAX; robust recovery from cliff termination. | `table_showcase1_mujoco.tex`<br>Fig. 3(a) |
| **Dataset 2A** | **TensorNEAT $\times$ QDax Framework Dissection**<br>Dynamic topological graph mutations (TensorNEAT) combined with structured MAP-Elites archives on XOR. | Illumination of 8/40 niches ($20.0\%$) vs. 1/40 for monolithic neuroevolution. | `table_showcase2_dissection.tex`<br>Fig. 3(b) |
| **Dataset 3A** | **2D Island Migration Staircase & Deme Scaling**<br>Ring-topology demes evaluated on 20D Rastrigin across island counts $M \in \{4, 16, 64\}$ ($P_{\text{total}} = 512$ to $8,192$). | $O(1)$ compilation latency invariance ($0.98\text{--}1.11\,\text{s}$) across scale. | `table_showcase4_islands_ablation.tex` |
| **Dataset 4A** | **Operator & Schedule Annealing Ablation**<br>Cosine-annealed vs. constant crossover/mutation schedules on 10D Rastrigin across 10 random seeds. | Cosine annealing achieves $2.89 \pm 2.41$ vs. $5.44 \pm 2.74$ (constant) and $13.19 \pm 10.88$ (mutation-only). | `table_showcase4_islands_ablation.tex` |
| **Dataset 5A** | **MAP-Elites Parity & Hardware Acceleration**<br>10D Rastrigin ($P=100$, 500 gens, 20 seeds). MalthusJAX Native vs. upstream QDax baseline. | **$1.68\times$ wall-clock speedup** ($0.073\,\text{ms/gen}$ vs $0.123\,\text{ms/gen}$) with strict statistical fitness parity ($p=0.202$). | `table_showcase3_qdax_speedup.tex`<br>Fig. 6(a)-(d) |

---

## 2. Hardware & Environment Requirements

### Recommended Cluster Environment (Target Platform)
* **GPU**: NVIDIA H100 NVL (94–96 GB VRAM) or A100 (40/80 GB VRAM).
* **Driver / CUDA**: Driver $\ge 535.0$, CUDA 12.0–12.8.
* **Host RAM**: $\ge 64$ GB per job.
* **OS**: Linux (Ubuntu 22.04 LTS / RHEL 9 / Rocky Linux).
* **Python Environment**: Python 3.10–3.12 with Conda (`post_sub` or `malthusjax`).

### Local Workstation (Validation Platform)
* **Apple Silicon (M1/M2/M3/M4, 16GB+ Unified Memory)** or **Linux PC with RTX 3080/4090**.
* JAX with Metal (`jax-metal`) or CPU backend for quick smoke-testing.

---

## 3. Autonomous Data Generation Commands

All benchmarks are driven by the unified script in the framework repository:
`examples/generate_empirical_paper_curves.py`.

### Slurm Cluster Batch Execution

The primary entry point for unattended, autonomous cluster generation is `scripts/submit_empirical_curves.sh`.

#### 1. Standard Publication Benchmark (Full Multi-Seed Run)
Executes the complete empirical suite across all seeds and environments (~1.5 hours on an H100):
```bash
# In the MalthusJAX repository root:
sbatch scripts/submit_empirical_curves.sh
```

#### 2. Smoke Validation Mode (Fast Pipeline Verification)
Executes all 5 pipelines on minimal horizons ($G=10$, $P=16$) to verify imports, JIT compilations, and output serialization in $\sim 30\text{--}45$ seconds:
```bash
sbatch scripts/submit_empirical_curves.sh --smoke
```

#### 3. Heavy Stress Benchmark Mode
Executes high-density stress testing with enlarged population demes and extended generation horizons:
```bash
sbatch scripts/submit_empirical_curves.sh --heavy
```

#### 4. Monitoring Slurm Jobs
```bash
# Check queue status:
squeue -u $USER

# Follow live output:
tail -f logs/mjx_paper_curves_<JOB_ID>.out

# Follow live error/warning logs:
tail -f logs/mjx_paper_curves_<JOB_ID>.err
```

---

### Direct Python Execution

If running interactively within a `tmux` session, on a dedicated compute node, or on a local workstation:

```bash
# 1. Activate your Conda/Venv environment:
conda activate post_sub

# 2. Source the required memory guards (CRITICAL - see Section 4):
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export MUJOCO_GL="egl"

# 3. Launch the empirical suite:
# Standard Publication Tier:
python examples/generate_empirical_paper_curves.py

# Quick Smoke Test:
python examples/generate_empirical_paper_curves.py --smoke

# Heavy Tier:
python examples/generate_empirical_paper_curves.py --heavy
```

---

## 4. XLA, JAX & Hardware Memory Optimization

Accelerated Linear Algebra (XLA) and JAX provide immense throughput by compiling computation graphs into fused GPU kernels. However, default XLA behavior can easily induce catastrophic Out-of-Memory (OOM) failures or host RAM exhaustion on shared nodes. 

Apply the following configurations:

### GPU VRAM: Taming Greedy Allocation

#### The Problem
By default, JAX attempts to preallocate **75% to 90% of total GPU VRAM** immediately upon the first JIT execution. On a 96 GB H100, this locks $\sim 86\text{ GB}$ of VRAM upfront. When secondary libraries (e.g., MuJoCo MJX CUDA buffers, CUDA runtime contexts, or PyTorch bridges) attempt to allocate device memory, the CUDA driver aborts with:
```
RuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate ... bytes.
```

#### The Solution
Configure dynamic, on-demand device allocation and cap the maximum memory pool:

```bash
# Disable greedy upfront preallocation:
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Optional: Cap memory fraction (e.g. 60% of GPU memory):
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
```

Inside Python, ensure these environment variables are set **before** importing `jax` or `jax.numpy`:
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

import jax
```

---

### Host RAM: Preventing Thread Pool Explosions

#### The Problem
High-performance compute nodes frequently feature 64 to 256 CPU hardware threads. When JAX, NumPy, SciPy, and OpenBLAS initialize, each library spawns a worker pool matching the host thread count (e.g., $128 \times 4 \approx 512$ threads). During JIT tracing, memory footprint explodes and CPU thread contention degrades throughput, often triggering Slurm `oom-killer` kills on the host node.

#### The Solution
Clamp all host BLAS and linear algebra thread pools to 1:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

---

### XLA Compilation Storms & Caching

#### The Problem
If loop bounds, population sizes, or parameter dictionaries change dynamically across generations or seeds, XLA treats each invocation as a distinct graph signature, triggering a new JIT compilation. Compiling hundreds of distinct HLO graphs consumes host RAM linearly until exhaustion.

#### The Solution
1. **Enforce Static Shapes**: All population dimensions ($P$), chromosome lengths ($D$), and generation counts ($G$) must remain static literals. Use `jax.lax.scan` for evolutionary loops rather than Python `for` loops.
2. **Persistent On-Disk Compilation Cache**: Cache compiled PTX binaries to disk so subsequent runs or restarts skip expensive compilation:
   ```bash
   export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_compilation_cache"
   export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
   ```
3. **Engine-Level Cache Clearing**: If executing multiple consecutive large benchmarks within a single long-lived Python process, clear intermediate JIT dispatch caches and trigger garbage collection between benchmarks:
   ```python
   import gc
   import jax

   # Between benchmark dataset executions:
   jax.clear_caches()
   gc.collect()
   ```

---

### Asynchronous Dispatch & Memory Accumulation

#### The Problem
JAX methods execute asynchronously: calls return immediately with `DeviceArray` futures while calculations queue up on the GPU stream. In a tight evaluation loop, Python can queue millions of device operations faster than the GPU can retire them, ballooning host memory with unresolved future metadata.

#### The Solution
Insert explicit synchronization barriers on timing checkpoints and epoch boundaries using `.block_until_ready()`:
```python
# Force GPU stream to drain before taking wall-clock measurements:
final_state = state.block_until_ready()
```

---

### Headless MuJoCo GPU Offscreen Rendering

When running MuJoCo MJX continuous control benchmarks (Dataset 1A) on headless cluster nodes lacking an active X11 display:
```bash
# Linux (NVIDIA EGL Hardware Rendering):
export MUJOCO_GL="egl"

# macOS (Apple Silicon CoreGraphics/Metal):
export MUJOCO_GL="cgl"
```
*Note: If the log outputs `Failed to import warp: No module named 'warp'`, this is harmless telemetry confirming that MuJoCo MJX successfully bypassed optional PyTorch/Warp plugins and used native JAX CUDA.*

---

## 5. Complete Verification & Test Suite Commands

Before launching extensive multi-seed benchmarks, verify the codebase integrity using the automated test suite.

### Running Unit & Integration Tests (Framework Repo)
```bash
cd /path/to/MalthusJAX

# 1. Run the entire test suite:
pytest -v

# 2. Run core execution engine tests (verifying lax.scan & JIT safety):
pytest -v tests/test_fast_engine.py

# 3. Verify Level 1 PRNG key splitting and zero-mutation determinism:
pytest -v tests/test_prng_keys.py

# 4. Verify genetic operators (crossover, mutation, selection):
pytest -v tests/test_operators.py

# 5. Verify fitness evaluators and environment adapters:
pytest -v tests/test_evaluators.py

# 6. Verify island models and migration topologies:
pytest -v tests/test_island_models.py
```

### Running Fast Sanity Check of the Generator
```bash
# Validates the full empirical generator end-to-end in < 1 minute:
python examples/generate_empirical_paper_curves.py --smoke
```

---

## 6. Data Synchronization & Paper Compilation Pipeline

Once the cluster job completes, synchronize the data back to the paper repository and recompile the manuscript:

### Step 1: Sync Cluster Results to Local Paper Repo
Run these commands from your local machine (adjust SSH host/user as configured):
```bash
cd /Users/leonardodicaterina/Documents/GitHub/MalthusJAX_Paper

# Fetch empirical curves JSON:
rsync -avzP 20240485@DAH2:~/post_submission/results/empirical_curves.json ./figures/data/

# Fetch generated LaTeX tables:
rsync -avzP 20240485@DAH2:~/post_submission/results/tables/ ./tables/
```

### Step 2: Regenerate All Paper Figures & Schemas
Update all vector figures (`figures/*.pdf`) using the freshly ingested JSON data:
```bash
python3 scripts/generate_paper_schemas.py
```
This script reads `figures/data/empirical_curves.json` and updates:
* `figures/fig3_showcases_control_and_dissection.pdf` (Continuous control & NEAT dissection)
* `figures/fig6_qdax_parity_quad_plot.pdf` (MAP-Elites 4-panel speedup & parity)
* Architectural and pipeline vector schemas (`fig1` through `fig8`).

### Step 3: Recompile the Manuscript PDF
Compile the publication manuscript using `latexmk`:
```bash
make
```
To clean auxiliary files and recompile from scratch:
```bash
make clean
make
```

### Step 4: Verify Strict 12-Page Budget
Check that the manuscript strictly satisfies the 12-page IEEE/ACM length constraint:
```bash
python3 -c "import pypdf; reader = pypdf.PdfReader('main.pdf'); print(f'Total pages: {len(reader.pages)} (Must be 12)')"
```

---

## 7. Troubleshooting & Diagnostic FAQ

### Q1: The job fails with `CUDA_ERROR_OUT_OF_MEMORY`.
* **Fix**: Ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false` is exported.
* Lower `XLA_PYTHON_CLIENT_MEM_FRACTION` from `0.60` to `0.45` if running alongside large MuJoCo MJX model geometries.
* Check for competing processes using `nvidia-smi`.

### Q2: The Slurm node abruptly cancels the job with `oom-killer` or `slurmstepd: error: Exceeded step memory limit`.
* **Fix**: This is **Host RAM exhaustion**, not GPU VRAM. Ensure thread pool clamping variables (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, etc.) are exported before launching Python.
* Increase Slurm host memory reservation: `#SBATCH --mem=64G` or `--mem=96G`.

### Q3: Why does `generate_paper_schemas.py` fail with `FileNotFoundError`?
* **Fix**: Ensure you have synced `empirical_curves.json` from the cluster to `figures/data/empirical_curves.json`. Run:
  ```bash
  mkdir -p figures/data
  cp ../MalthusJAX/results/empirical_curves.json figures/data/
  ```

### Q4: Bitwise Dual-Parity discrepancy in Dataset 1A.
* **Explanation**: MalthusJAX guarantees bitwise float32 parity when using identical PRNG seeds and identical operator pipelines (`malthusjax_evosax` vs `evosax_open_es`). If discrepancies occur:
  1. Verify both algorithms use the identical seed.
  2. Confirm `JAX_ENABLE_X64=False` is uniformly set (or unset) across both implementations.
  3. Ensure no non-deterministic atomic operations or fast-math approximations (`--ffast-math`) are injected by external compilers.

---

*Authored for the MalthusJAX Research Group. Maintained in `MalthusJAX_Paper/DATA_REPRODUCTION_GUIDE.md`.*
