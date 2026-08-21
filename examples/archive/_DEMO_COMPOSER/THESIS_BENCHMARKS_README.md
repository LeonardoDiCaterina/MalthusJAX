# Thesis Benchmarking Suite

Automated experiment runner for thesis experiments with reproducible results and high-quality plots.

## Experiments Included

### Convergence Studies (100 seeds each, 5 pipelines)

1. **convergence_sphere_dim10.toml**
   - Sphere function, 10 dimensions
   - Baseline for optimal convergence behavior

2. **convergence_sphere_dim20.toml**
   - Sphere function, 20 dimensions
   - Scalability test

3. **convergence_rosenbrock_dim10.toml**
   - Rosenbrock (valley) function
   - Tests navigation through curved landscapes

4. **convergence_ellipsoidal_dim10.toml**
   - Ellipsoidal rotated function
   - Tests ill-conditioned optimization

### Pipelines Tested

Each experiment compares:
- **malthusjax_default**: Elite pool selection (baseline)
- **malthusjax_roulette**: Roulette selection variant
- **malthusjax_tournament**: Tournament selection variant
- **evosax_simplega**: Evosax SimpleGA reference
- **evosax_differential_evolution**: Evosax DE reference

## Quick Start

### 1. Run All Experiments (Background)

```bash
# Start all experiments with nohup
bash run_thesis_experiments.sh

# Monitor progress
tail -f thesis_bench.log

# Or run directly
python run_all_thesis_experiments.py
```

### 2. Run Specific Experiment

```bash
python run_all_thesis_experiments.py --single sphere_dim10
```

### 3. Quick Test (No Plots)

```bash
python run_all_thesis_experiments.py --skip-plots
```

## Output Structure

Each experiment generates:

```
results/thesis/
├── convergence_sphere_dim10/
│   ├── convergence_seeds_0-3.png        (4 seed overlays)
│   ├── timing_boxplot.png               (execution time distribution)
│   ├── final_best_fitness_boxplot.png   (final fitness robustness)
│   ├── summary_table.tex                (LaTeX table)
│   ├── aggregated_summary.json          (all statistics)
│   └── convergence_seed_0.json          (full history seed 0)
├── convergence_sphere_dim20/
│   └── ...
├── convergence_rosenbrock_dim10/
│   └── ...
└── convergence_ellipsoidal_dim10/
    └── ...
```

## For Thesis Writing

### Generate Summary Table

```python
import json
from pathlib import Path

# Load results
results_dir = Path("results/thesis")
for exp_dir in results_dir.glob("*/"):
    with open(exp_dir / "aggregated_summary.json") as f:
        data = json.load(f)
    # Use for thesis tables
```

### Use LaTeX Tables

Copy `.tex` files directly into thesis:

```latex
\begin{table}
  \input{results/thesis/convergence_sphere_dim10/summary_table}
  \caption{Convergence results on Sphere function (dim=10)}
\end{table}
```

### Include Plots

All plots are high-resolution (300 DPI) with optimized x-axis rotation:

```latex
\begin{figure}
  \includegraphics[width=0.9\textwidth]{results/thesis/convergence_sphere_dim10/convergence_seeds_0-3.png}
  \caption{Convergence curves across 4 seeds}
\end{figure}
```

## Performance Notes

### Expected Runtime

- **convergence_sphere_dim10**: ~30 min (100 seeds × 5 pipelines)
- **convergence_sphere_dim20**: ~60 min (higher dimensionality)
- **convergence_rosenbrock_dim10**: ~40 min (slower convergence expected)
- **convergence_ellipsoidal_dim10**: ~50 min

**Total**: ~3-4 hours for all 4 experiments

### Resource Requirements

- GPU: NVIDIA GPU recommended (JAX/XLA compilation)
- CPU: Multi-core (parallel seed execution)
- Memory: 16GB+ recommended
- Disk: ~500MB for all results

## Configuration Details

### Standard Settings

```toml
pop_size = 100          # Population size
generations = 500       # Per-seed generations
seeds = 100             # Reproducibility
selection = elite_pool  # Default for MalthusJAX
mutation_rate = 0.1     # Gaussian mutation
```

### PRNG & Reproducibility

- Fixed seeds: 42-141 (100 seeds)
- Shared initialization: `pop_seed=123`
- JAX key splitting: Fixed per experiment

## Troubleshooting

### OutOfMemoryError

Reduce `pop_size` in TOML or run fewer experiments.

### Slow Execution

Check GPU utilization: `nvidia-smi` should show >80%

### Missing Results

Check `thesis_bench.log` for error messages. Re-run single experiment:
```bash
python run_all_thesis_experiments.py --single sphere_dim10
```

## Next Steps for Thesis

1. Run experiments with nohup
2. While running, write Methodology section (test harness, TOML configs)
3. After completion:
   - Export summary tables
   - Embed high-res plots
   - Write Results section with interpretations
4. Cross-reference: Each result → methodology section explaining it

---

**Created for thesis:**
- Reproducible experimental framework
- High-quality publication-ready plots
- Automated data export for writing
