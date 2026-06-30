# MalthusJAX Environment & Data Topology

This document provides a practical, up-to-date map of the project's data architecture, detailing exactly what we have, how the pipelines are structured, and how the Local Mac and DAH2 Cluster environments interact.

---

## 1. Environment Roles

### The Cluster (`DAH2` - `20240485@10.10.80.4`)
* **Purpose:** Heavy lifting. Executes massive LHS and Cartesian grids across GPUs, producing thousands of JSON trace files. 
* **Analysis:** The `benchmark_analyzer.py` is run *here* to condense massive JSON traces into small `.csv` tables and `.png` plots.
* **Environment:** `conda activate mjx_gpu_clean`

### The Local Machine (Your Mac)
* **Purpose:** Code authoring, smoke testing, and thesis writing.
* **Data Policy:** **NEVER** pull raw JSON traces to the local machine. We only pull the `analysis/` folders containing the final regression tables and plots.
* **Environment:** `conda activate GP_env_2`

---

## 2. Practical Data Topology (What We Actually Have)

If you look inside the `results/` folder, you'll see two distinct types of directories: **Pipelines** (the algorithms) and **Experiment Suites** (the execution outputs).

### The Pipelines (Algorithm Configurations)
These directories store the raw output traces for each specific algorithm configuration.
* `evosax_baseline`: The pure, independent EvoSAX execution.
* `malthusjax_wrapper`: MalthusJAX engine wrapping EvoSAX functions (The Parity Control).
* `mjx_baseline`: MalthusJAX engine with native operators.
* **Ablation Variants:** `mjx_ablate_mutation`, `mjx_ablate_crossover`, `mjx_ablate_sel_elite`, `mjx_ablate_sel_tournament`.
* **Precision Variants:** `mjx_bf16`, `mjx_f16`, `mjx_f32`.

### The Experiment Suites (The Analyzed Outputs)
These are the overarching benchmark folders. When `benchmark_analyzer.py` is run, it looks across the pipelines and deposits the final regression tables and plots into the `analysis/` subfolder of these suites.
* `h1_parity`: The standard grid equivalence tests (Sphere, Rosenbrock, Rastrigin).
* `h2_ablation_hard`: The structural ablation tests executed on the Hard Mode BBOB landscapes (Lunacek, Schwefel, Gallagher) using Latin Hypercube Sampling (LHS).
* `h3_representation_hard`: The precision downcasting tests (`float32` vs `bfloat16`/`float16`) on the Hard Mode landscapes.
* `*_smoke`: Various smoke-test directories (e.g., `h2_ablation_hard_smoke`). These are tiny, low-generation runs executed locally just to verify no code crashes before deploying to the cluster.

---

## 3. The Synchronization Protocol (Cluster ⇄ Local)

Whenever a benchmark suite finishes on the cluster, you must generate the analysis on the cluster, and then use `scp` to pull *only* the `analysis/` folder down to your Mac.

### Step 1: Run the Analyzer on the Cluster
SSH into DAH2 and run the analyzer for the specific suite you just completed:
```bash
# Example for H1
python scripts/benchmark_analyzer.py --toml configs/h1_parity_lhs.toml --data_dir results/h1_parity

# Example for H2 Hard
python scripts/benchmark_analyzer.py --toml configs/h2_ablation_hard_lhs.toml --data_dir results/h2_ablation_hard

# Example for H3 Hard
python scripts/benchmark_analyzer.py --toml configs/h3_representation_hard_lhs.toml --data_dir results/h3_representation_hard
```

### Step 2: Sync the Analysis Down to Local
Run these commands from your **Local Mac Terminal** to pull the plots and OLS `.csv` tables:

```bash
# Sync H1 Parity Analysis
scp -r 20240485@10.10.80.4:~/test_mjx/MalthusJAX/results/h1_parity/analysis /Users/leonardodicaterina/Documents/GitHub/MalthusJAX/results/h1_parity/

# Sync H2 Ablation (Hard Mode) Analysis
scp -r 20240485@10.10.80.4:~/test_mjx/MalthusJAX/results/h2_ablation_hard/analysis /Users/leonardodicaterina/Documents/GitHub/MalthusJAX/results/h2_ablation_hard/

# Sync H3 Representation (Hard Mode) Analysis
scp -r 20240485@10.10.80.4:~/test_mjx/MalthusJAX/results/h3_representation_hard/analysis /Users/leonardodicaterina/Documents/GitHub/MalthusJAX/results/h3_representation_hard/
```

### Step 3: Thesis Integration
Once the `analysis/` folders are local, the `.png` plots can be copied directly into `thesis_chapters/images/`, and the OLS results can be written directly into the markdown draft.
