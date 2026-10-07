# Malthus-Optimize: JAX-Accelerated SciPy Optimization Powered by Composer

[![JAX](https://img.shields.io/badge/Backend-JAX%20%7C%20Flax-blue.svg)](https://github.com/google/jax)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Engine: Composer](https://img.shields.io/badge/Engine-MalthusJAX%20Composer-purple.svg)](../../src/malthusjax/composer)

**`malthus-optimize`** is a high-throughput, GPU/TPU-accelerated, drop-in replacement for Python's standard `scipy.optimize` package. Built on Google JAX and orchestrated directly by the **MalthusJAX Composer Engine**, it delivers up to **10–100× speedups** on continuous multimodal global optimization and non-linear curve fitting while maintaining 100% API compatibility with `scipy.optimize`.

---

## Highlights

- **100% SciPy Drop-In Compatibility**: Seamlessly replace `from scipy.optimize import differential_evolution, minimize, curve_fit` with `from malthus_optimize import ...` with zero application code changes.
- **Powered by MalthusJAX Composer**: Uses Composer's `UniversalAdapterEngine` and EvoSAX integration (`DifferentialEvolution`, `CMA_ES`, `PSO`, `SimpleGA`, `OpenES`).
- **Zero Python Overhead during Search**: The full evolutionary loop compiles into static XLA scan tapes (`jax.lax.scan`), evaluating entire generations in a single fused GPU kernel.
- **Dual-Mode Objective Evaluation**:
  - *Pure JAX Mode*: Fully vectorized objective functions compiled with `@jax.jit` and `jax.vmap`.
  - *Host Fallback Mode*: Black-box NumPy, SciPy, or C-extension functions executed safely via `jax.pure_callback(..., vmap_method="sequential")` inside the scan tape without tracer errors.
- **Global Non-Linear Curve Fitting**: `curve_fit` utilizes global evolutionary search, eliminating sensitivity to fragile initial parameter guesses (`p0`) and returning rigorous parameter covariance matrices ($P_{\text{cov}}$).
- **Embedded `mopt` CLI**: Benchmarking and command-line optimization suite ready out-of-the-box.

---

## Architectural Overview

```mermaid
graph TD
    subgraph SciPy_API["1. SciPy-Compatible Facade Layer"]
        DE["differential_evolution(...)"]
        MIN["minimize(method=...)"]
        CF["curve_fit(...)"]
        OPT_RES["OptimizeResult (dict + properties)"]
    end

    subgraph Search_Domain["2. Domain Normalization"]
        PARSE["parse_bounds(bounds, dim)"]
        HYPER["Unit Hypercube: [0, 1]^D"]
        DENORM["Denormalizer: x = lb + u * (ub - lb)"]
    end

    subgraph Evaluator_Bridge["3. Dual-Mode Evaluators"]
        PURE["Pure JAX Traced Path<br/>(@jax.jit + jax.vmap)"]
        CALLBACK["Host Fallback Bridge<br/>(jax.pure_callback)"]
        LS["LeastSquaresEvaluator<br/>(Weighted Residual Sum of Squares)"]
    end

    subgraph Composer_Runtime["4. MalthusJAX Composer Engine"]
        ADAPTER["build_evosax_engine(...)"]
        STRAT["EvoSAX Strategies<br/>(DE, CMA-ES, PSO, GA)"]
        SCAN["XLA Scan Loop Tape<br/>(jax.lax.scan)"]
    end

    DE --> PARSE
    MIN --> PARSE
    CF --> LS
    LS --> PARSE
    PARSE --> HYPER

    HYPER --> Evaluator_Bridge
    PURE --> ADAPTER
    CALLBACK --> ADAPTER

    ADAPTER --> STRAT
    STRAT --> SCAN
    SCAN --> DENORM
    DENORM --> OPT_RES
```

---

## Performance & Feature Matrix vs. SciPy

| Capability / Metric | `scipy.optimize` | `malthus-optimize` |
| :--- | :---: | :---: |
| **API Compatibility** | Reference Standard | **Drop-In Identical** |
| **Compute Acceleration** | CPU Only (Single-Core) | **CPU, Apple Silicon (Metal), NVIDIA GPU, TPU** |
| **Execution Engine** | Python `while` loops | **Static XLA Tapes (`jax.lax.scan`)** |
| **Population Vectorization** | Serial / Python multiprocessing | **SIMD / SIMT Native Kernel Parallelism** |
| **Initial Guess Sensitivity (`curve_fit`)** | High (Trapped in local minima) | **Zero (Global Evolutionary Search)** |
| **Available Metaheuristics** | Differential Evolution, Basinhopping | **DE, CMA-ES, Particle Swarm (PSO), Genetic Algorithm (GA), OpenES** |
| **Full Parameter Covariance ($P_{\text{cov}}$)** | Yes | **Yes (Finite-Difference Numerical Jacobian)** |
| **Memory Footprint** | Object allocations per candidate | **Constant Pre-Allocated JAX Static Tensors** |

---

## Showcase Gallery

### 1. Damped Harmonic Oscillator Curve Fitting
Non-linear parameter estimation fitting 5 physical parameters ($A, \gamma, \omega, \phi, C$) of a damped sinusoidal signal in 1.5 seconds with exact covariance quantification, overcoming local minima traps that hinder gradient descent.

![Damped Oscillator Curve Fitting](examples/plots/01_damped_oscillator_fitting.png)

### 2. Multimodal Benchmark Landscape Trajectories
Differential Evolution, CMA-ES, Particle Swarm Optimization, and Genetic Algorithm navigating 5D Rastrigin and 5D Rosenbrock landscapes.

![Global Optimization Benchmarks](examples/plots/02_global_optimization_benchmarks.png)

### 3. 2D Sensor Network Trilateration
Geometric non-linear least squares estimating the unknown 2D coordinates of an emitter from noisy distance measurements to anchor stations in under 1 second.

![2D Sensor Network Trilateration](examples/plots/03_sensor_trilateration.png)

---

## Installation

```bash
# Editable install within workspace
pip install -e packages/malthus_optimize

# With developer testing dependencies
pip install -e "packages/malthus_optimize[dev]"
```

---

## Quick Start

### 1. Differential Evolution Drop-In

```python
import numpy as np
from malthus_optimize import differential_evolution

# Multimodal Rastrigin function
def rastrigin(x):
    return 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

bounds = [(-5.12, 5.12)] * 10

# Exactly matches SciPy call signature
res = differential_evolution(
    func=rastrigin,
    bounds=bounds,
    popsize=20,
    maxiter=100,
    seed=42,
)

print(f"Optimal Value: {res.fun:.6e}")
print(f"Best Parameters: {res.x}")
print(f"Total Evaluations: {res.nfev}")
```

### 2. Multi-Strategy `minimize` (CMA-ES, PSO, GA)

```python
from malthus_optimize import minimize

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

# Run CMA-ES on 5D Rosenbrock
res = minimize(
    fun=rosenbrock,
    method="cmaes",
    bounds=[(-5.0, 5.0)] * 5,
    options={"pop_size": 30, "max_generations": 150, "seed": 42},
)

print("CMA-ES Solution:", res.x)
print("Minimum Loss:   ", res.fun)
```

### 3. Non-Linear Curve Fitting with Uncertainty Covariance

```python
import numpy as np
from malthus_optimize import curve_fit

# Non-linear model
def model(t, a, b, c):
    return a * np.exp(-b * t) + c

tt = np.linspace(0, 4, 50)
yy = model(tt, 2.5, 1.3, 0.5) + np.random.normal(0, 0.05, size=len(tt))

# Fit parameters with bounded global search
popt, pcov = curve_fit(
    model,
    tt,
    yy,
    bounds=([0.0, 0.0, 0.0], [5.0, 5.0, 2.0]),
    seed=42,
)

perr = np.sqrt(np.diag(pcov))
print(f"Fitted a: {popt[0]:.3f} ± {perr[0]:.3f}")
print(f"Fitted b: {popt[1]:.3f} ± {perr[1]:.3f}")
print(f"Fitted c: {popt[2]:.3f} ± {perr[2]:.3f}")
```

---

## Command-Line Interface (`mopt`)

`malthus-optimize` provides the `mopt` CLI for benchmarking and running optimization tasks:

```bash
# Run comparative benchmark vs. scipy.optimize across test functions
mopt benchmarks --dim 10 --popsize 20 --maxiter 100

# Execute an individual optimization run
mopt run --function rastrigin --method cmaes --dim 5 --maxiter 80
```

---

## Documentation & User Guides

Explore comprehensive documentation and step-by-step guides in [`docs/`](docs/):

1. **[Architecture Specification](docs/architecture.md)**: Deep dive into the Composer bridge, dual-mode evaluation, and finite-difference Jacobian covariance computation.
2. **[01. SciPy Drop-In Migration](docs/guides/01_scipy_drop_in_migration.md)**: Zero-friction migration guide from legacy `scipy.optimize` codebases.
3. **[02. Differential Evolution Tuning](docs/guides/02_differential_evolution.md)**: Hyperparameters, mutation scale dithering, and boundary enforcement.
4. **[03. Multi-Strategy Optimization](docs/guides/03_multi_strategy_minimize.md)**: Guide to CMA-ES, PSO, GA, and OpenES selection.
5. **[04. Scientific Curve Fitting](docs/guides/04_scientific_curve_fitting.md)**: Complex peak deconvolution, spectroscopic fitting, and covariance error quantification.
6. **[05. Massively Parallel & Island Models](docs/guides/05_island_distributed_opt.md)**: Scaling with vector sweeps and MalthusJAX island topologies.
7. **[06. CLI & Configuration Workflows](docs/guides/06_cli_and_toml_workflows.md)**: Complete reference for the `mopt` command-line utility.

---

## License

This package is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
