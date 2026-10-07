# User Guide 4: Non-Linear Curve Fitting & Residual Uncertainty

`malthus_optimize.curve_fit` is a drop-in replacement for `scipy.optimize.curve_fit` designed for continuous scientific parameter estimation, time-series modeling, signal processing, and experimental curve fitting.

---

## 1. Why Evolutionary Curve Fitting?

Traditional Levenberg-Marquardt or Gauss-Newton curve fitters (`scipy.optimize.curve_fit(method='lm')`) rely on local gradient descent. When fitting:
- Oscillatory signals with unknown frequency or phase (e.g. damped sinusoids, acoustic decays)
- Multi-modal response curves or non-linear sigmoids
- Coupled multi-parameter physical equations

Local gradient descent frequently becomes trapped in sub-optimal local minima unless an initial guess `p0` extremely close to the true parameters is provided.

**`malthus-optimize.curve_fit`** uses global evolutionary search (`'de'` by default) across bounded parameter domains, eliminating the need for delicate manual initialization.

---

## 2. API Signature & Parameter Bounds

```python
popt, pcov = curve_fit(
    f,                  # Callable model: f(x, *params)
    xdata,              # Independent variables: 1D array or 2D (N, D)
    ydata,              # Dependent observations: 1D array
    p0=None,            # Optional initial guess
    sigma=None,         # Measurement uncertainties for weighted regression
    absolute_sigma=False, # Covariance scaling flag
    bounds=None,        # Parameter boundaries (list of tuples or 2-tuple of arrays)
    method="de",        # 'de', 'cmaes', 'pso', or 'ga'
    max_generations=100,# Optimization generation budget
    pop_size=30,        # Population size
    seed=42,            # Random seed
)
```

---

## 3. Working Example: Damped Harmonic Oscillator

```python
import numpy as np
import matplotlib.pyplot as plt
from malthus_optimize import curve_fit

# Damped harmonic oscillator model: y(t) = A * exp(-gamma * t) * cos(omega * t + phi) + C
def damped_oscillator(t, amplitude, decay_rate, omega, phase, offset):
    return amplitude * np.exp(-decay_rate * t) * np.cos(omega * t + phase) + offset

# Synthesize noisy experimental signal
t = np.linspace(0.0, 8.0, 120)
true_params = [5.0, 0.4, 4.5, 0.8, 1.2]
y_clean = damped_oscillator(t, *true_params)
y_obs = y_clean + np.random.normal(0.0, 0.25, size=len(t))

# Define bounded search domain (NO sensitive p0 guess needed!)
bounds = (
    [1.0, 0.05, 1.0, -np.pi, -2.0],  # Lower bounds
    [10.0, 1.5, 8.0, np.pi, 4.0],    # Upper bounds
)

popt, pcov = curve_fit(damped_oscillator, t, y_obs, bounds=bounds, seed=42)

# Extract standard errors from covariance diagonal
perr = np.sqrt(np.diag(pcov))

print(f"Fitted Amplitude : {popt[0]:.3f} ± {perr[0]:.3f} (True: {true_params[0]})")
print(f"Fitted Decay Rate: {popt[1]:.3f} ± {perr[1]:.3f} (True: {true_params[1]})")
print(f"Fitted Frequency : {popt[2]:.3f} ± {perr[2]:.3f} (True: {true_params[2]})")
```

---

## 4. Covariance & Parameter Confidence

The returned `pcov` is the estimated covariance matrix of `popt`.
- **Standard Errors**: $\sigma_{\mathbf{p}} = \sqrt{\text{diag}(P_{\text{cov}})}$
- **Parameter Correlation**:
  $$R_{i, j} = \frac{P_{\text{cov}}[i, j]}{\sqrt{P_{\text{cov}}[i, i] \cdot P_{\text{cov}}[j, j]}}$$
This provides rigorous scientific uncertainty quantification identical to SciPy's output.
