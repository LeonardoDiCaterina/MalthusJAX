# Thesis Benchmarking - Quick Commands

## Step 1: Start Experiments (Takes 3-4 hours)

Run in background so you can write thesis while benchmarks execute:

```bash
cd ~/Documents/GitHub/MalthusJAX/examples/_DEMO_COMPOSER

# Option A: Using bash launcher (recommended)
bash run_thesis_experiments.sh
# Then: tail -f thesis_bench.log

# Option B: Direct python with nohup
nohup python -u run_all_thesis_experiments.py > thesis_bench.log 2>&1 &

# Option C: Just run directly (in terminal, blocks)
python run_all_thesis_experiments.py
```

## Step 2: Monitor Progress (While Experiments Run)

```bash
# Watch the log in real-time
tail -f thesis_bench.log

# Or just check current status periodically
tail -20 thesis_bench.log

# Kill experiments if needed
kill <PID>  # from nohup output or ps aux
```

## Step 3: Once Complete - Write Your Thesis

### Methodology Section (Write While Running)

Reference these files:
- `THESIS_BENCHMARKS_README.md` - Test harness explanation
- `convergence_sphere_dim10.toml` - Example TOML config
- `results/thesis/*/summary_table.tex` - Algorithm details table

### Results Section (After Experiments Complete)

```bash
# View all results
ls -la results/thesis/*/

# Copy plots to thesis directory
cp results/thesis/convergence_sphere_dim10/convergence_seeds_0-3.png ~/thesis_figures/

# Copy LaTeX tables
cat results/thesis/convergence_sphere_dim10/summary_table.tex
```

## File Structure Created

```
examples/_DEMO_COMPOSER/
├── convergence_sphere_dim10.toml          ← Run #1
├── convergence_sphere_dim20.toml          ← Run #2
├── convergence_rosenbrock_dim10.toml      ← Run #3
├── convergence_ellipsoidal_dim10.toml     ← Run #4
├── run_all_thesis_experiments.py          ← Master runner
├── run_thesis_experiments.sh              ← nohup launcher
└── THESIS_BENCHMARKS_README.md            ← Full documentation

results/thesis/
├── convergence_sphere_dim10/
│   ├── convergence_seeds_0-3.png          ← Use in results
│   ├── timing_boxplot.png                 ← Use in results
│   ├── final_best_fitness_boxplot.png     ← Use in results
│   ├── summary_table.tex                  ← Use in results
│   └── aggregated_summary.json            ← Statistics
├── convergence_sphere_dim20/
│   └── ...
├── convergence_rosenbrock_dim10/
│   └── ...
└── convergence_ellipsoidal_dim10/
    └── ...
```

## Expected Runtime Breakdown

- **convergence_sphere_dim10**: 30 min
- **convergence_sphere_dim20**: 60 min  
- **convergence_rosenbrock_dim10**: 40 min
- **convergence_ellipsoidal_dim10**: 50 min
- **Total**: ~3.5 hours (sequential)

## Key Files for Thesis

### ✓ For Methodology Section
- `convergence_sphere_dim10.toml` → Explain TOML-driven configuration
- `run_all_thesis_experiments.py` → Document test harness automation
- `src/malthusjax/benchmarking/README.md` → Reference BenchmarkRunner architecture

### ✓ For Results Section
Use these directly:
```
results/thesis/convergence_sphere_dim10/convergence_seeds_0-3.png
results/thesis/convergence_sphere_dim10/timing_boxplot.png
results/thesis/convergence_sphere_dim10/final_best_fitness_boxplot.png
results/thesis/convergence_sphere_dim10/summary_table.tex
results/thesis/convergence_sphere_dim20/
results/thesis/convergence_rosenbrock_dim10/
results/thesis/convergence_ellipsoidal_dim10/
```

## One-Line Commands

```bash
# Run everything with logging
cd ~/Documents/GitHub/MalthusJAX/examples/_DEMO_COMPOSER && nohup python -u run_all_thesis_experiments.py > thesis_bench.log 2>&1 &

# Monitor
tail -f ~/Documents/GitHub/MalthusJAX/examples/_DEMO_COMPOSER/thesis_bench.log

# View all results when done
ls -la ~/Documents/GitHub/MalthusJAX/examples/_DEMO_COMPOSER/results/thesis/*/
```

---

## Writing Order (Recommended)

1. **While experiments run** (3-4 hours)
   - Write Methodology: Overview, Test Harness, Experimental Design
   - Reference TOML configs and framework documentation

2. **After experiments complete**
   - Export results: Copy .png and .tex files to thesis folder
   - Write Results: Reference the generated plots and tables
   - Write Discussion: Compare findings

3. **Final pass**
   - Verify all plot captions match methodology
   - Ensure table citations are consistent
   - Update references section with code/framework citations
