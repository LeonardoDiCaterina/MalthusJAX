# %% [markdown]
# # Dynamic Benchmarking in MalthusJAX
#
# MalthusJAX ships with a powerful Benchmarking Engine capable of generating
# rigorous statistical comparisons across dimensionalities, problems, and seeds.
#
# Crucially, the benchmark runner is **Metric Agnostic**. It will dynamically serialize
# any scalar metric emitted by any `EngineAdapter`, and the analyzer will automatically
# generate OLS Regressions and Boxplots for those exact metrics.
#
# In this notebook, we'll demonstrate this by programmatically creating a TOML configuration,
# running a multi-seed grid search on QDAX, and running the analyzer.

# %%
import os
import sys
from pathlib import Path

# If running as a notebook in examples/showcase, change directory to the repository root
if Path(os.getcwd()).name == "showcase":
    os.chdir("../..")
print(f"Working Directory: {os.getcwd()}")

import subprocess

import toml

# %% [markdown]
# ## 1. Defining the TOML Configuration
#
# The `benchmark_runner` is driven entirely by `.toml` files. We define the `grid`
# (problems and dimensionalities), the `analysis` targets (what metrics to track),
# and the `pipelines` (which adapters to use).
#
# We'll set up a tiny grid just for demonstration purposes so it runs quickly.

# %%
config_dict = {
    "suite": {
        "name": "demo_benchmark",
        "mode": "cartesian",
        "output_dir": "results/demo_benchmark",
        "num_seeds": 10,
    },
    "grid": {
        "functions": ["rosenbrock", "rastrigin"],
        "dims": [5],  # Keep it small!
        "pops": [128],
        "gens": [20],
    },
    "analysis": {
        "reference_pipeline": "qdax_baseline",
        # Notice we are tracking QDAX specific metrics here!
        "target_metrics": ["best_fitness", "qd_score", "coverage", "execution_time"],
    },
    "pipelines": {
        "qdax_baseline": {
            "backend": "qdax",
            "qdax_strategy": "MAPElites",
            "qdax_num_descriptors": 2,
            "qdax_num_centroids": 100,
            "qdax_mutation_sigma": 0.1,
            "maximize": False,
        },
        "native_ga": {
            "backend": "malthusjax",
            "selection": "tournament:tournament_size=3",
            "crossover": "uniform_real:crossover_rate=0.5",
            "mutation": "gaussian:mutation_rate=0.1",
        },
    },
}

os.makedirs("results/showcase", exist_ok=True)
toml_path = Path("results/showcase/demo_benchmark.toml")

with open(toml_path, "w") as f:
    toml.dump(config_dict, f)

print(f"Saved benchmark configuration to {toml_path}")

# %% [markdown]
# ## 2. Running the Benchmark Suite
#
# We'll use the programmatic entry point to `scripts/benchmark_runner.py`.
# This will execute 2 functions * 1 dims * 3 seeds = 6 independent experiments.

# %%
output_dir = Path("results/demo_benchmark")

print("Starting benchmark suite...")
try:
    subprocess.run(
        [sys.executable, "scripts/benchmark_runner.py", "--toml", str(toml_path)],
        check=True,
        capture_output=True,
        text=True,
    )
except subprocess.CalledProcessError as e:
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    raise e
print(f"Benchmark data flushed to {output_dir}")

# %% [markdown]
# ## 3. Analyzing the Results Dynamically
#
# Now we pass the data directory to `scripts/benchmark_analyzer.py`.
# Because we specified `["best_fitness", "qd_score", "coverage", "execution_time"]` in the TOML,
# the analyzer will automatically synthesize LaTeX tables, CSVs, and PNG boxplots for *every single one* of those metrics.
#
# *Note: The Native GA does not track `qd_score` or `coverage`. The analyzer handles these missing columns gracefully by dropping NaNs during plot generation!*

# %%
print("Starting dynamic analyzer...")
subprocess.run(
    [
        sys.executable,
        "scripts/benchmark_analyzer.py",
        "--toml",
        str(toml_path),
        "--data_dir",
        str(output_dir),
    ],
    check=True,
)
print("Analysis complete!")

# %% [markdown]
# ## 4. Inspecting the Generated Artifacts
#
# Let's list the files generated in the analysis directory to prove it generated plots for the dynamic metrics!

# %%
analysis_dir = output_dir / "analysis"
artifacts = list(analysis_dir.glob("*"))

print("\nGenerated Artifacts:")
for artifact in sorted(artifacts):
    print(f" - {artifact.name}")

# %% [markdown]
# You'll notice artifacts like `native_ga_vs_qdax_baseline_rosenbrock_qd_score_scaling.png` exist, even though `qd_score` is entirely bespoke to QDAX!
#
# MalthusJAX is fully open-closed: to add a new metric to your reports, just have your adapter return it, and add its name to `target_metrics` in the TOML. No analyzer logic changes required!
