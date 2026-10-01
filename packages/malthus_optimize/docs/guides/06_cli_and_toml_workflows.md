# User Guide 6: CLI & Configuration Workflows (`mopt`)

`malthus-optimize` includes a lightweight, zero-dependency CLI tool called `mopt`. It allows benchmarking performance against `scipy.optimize` and running immediate command-line optimization tasks.

---

## 1. Installation & Verification

Once `malthus-optimize` is installed in your Python environment:

```bash
# Verify installation and view available subcommands
mopt --help
```

Output:
```text
usage: mopt [-h] {benchmarks,run} ...

Malthus-Optimize CLI: SciPy-compatible optimization powered by JAX.

positional arguments:
  {benchmarks,run}  Available subcommands
    benchmarks      Run comparative benchmarks vs scipy.optimize
    run             Run optimization on a benchmark function
```

---

## 2. Comparing Speeds: `mopt benchmarks`

The `benchmarks` command runs side-by-side comparative trials between `malthus_optimize.differential_evolution` (JAX/Composer) and `scipy.optimize.differential_evolution` (CPU).

```bash
# Run 10-dimensional benchmark across all standard functions
mopt benchmarks --dim 10 --popsize 20 --maxiter 100 --seed 42
```

Sample Output:
```text
===========================================================================
  Malthus-Optimize Benchmark Suite (dim=10, popsize=20, maxiter=100)
===========================================================================

--- Benchmark Function: SPHERE ---
Malthus-Optimize (JAX/Composer):
  Best Value: 1.458291e-08
  Execution Time: 0.1420 s (2000 evaluations)
SciPy (Sequential CPU):
  Best Value: 2.193021e-08
  Execution Time: 0.7850 s (2000 evaluations)
  Speedup: 5.53x

--- Benchmark Function: RASTRIGIN ---
Malthus-Optimize (JAX/Composer):
  Best Value: 0.000000e+00
  Execution Time: 0.1650 s (2000 evaluations)
SciPy (Sequential CPU):
  Best Value: 1.989918e-01
  Execution Time: 1.1240 s (2000 evaluations)
  Speedup: 6.81x
```

### Filtering Specific Test Functions
```bash
# Target only the Rosenbrock banana valley
mopt benchmarks --function rosenbrock --dim 5 --maxiter 150
```

---

## 3. Immediate Execution: `mopt run`

You can run individual optimization trials directly from your shell:

```bash
# Solve 5-dimensional Rastrigin with CMA-ES
mopt run --function rastrigin --method cmaes --dim 5 --maxiter 100

# Solve Ackley with Particle Swarm Optimization
mopt run --function ackley --method pso --dim 8 --popsize 30 --maxiter 120
```

---

## 4. Scripting with Shell Pipelines

Because `mopt` outputs standard human-readable text and clean return codes (`0` for success, non-zero for error), it can be embedded into shell scripts, CI/CD health checks, or benchmark tracking harnesses:

```bash
#!/usr/bin/env bash
set -e

for METHOD in de cmaes pso ga; do
  echo "Evaluating method: $METHOD"
  mopt run --function sphere --method "$METHOD" --dim 10 --maxiter 50
done
```
