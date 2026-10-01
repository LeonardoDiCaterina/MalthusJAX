# User Guide 4: Scientific Non-Linear Curve Fitting & Residual Analysis

`malthus_optimize.curve_fit` is a drop-in replacement for `scipy.optimize.curve_fit` designed specifically for experimental physics, spectroscopy, chemical kinetics, and materials science.

---

## 1. Why Evolutionary Curve Fitting?

Traditional Levenberg-Marquardt or Gauss-Newton curve fitters (`scipy.optimize.curve_fit(method='lm')`) rely on local gradient descent. When fitting:
- Overlapping spectroscopic peaks (e.g. XRD doublets, NMR, Raman, XPS)
- Multi-exponential decay curves
- Highly non-linear Arrhenius or Michaelis-Menten dynamics

Local optimizers frequently get trapped in sub-optimal local minima unless an initial guess `p0` within 1% of the true parameters is provided.

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

## 3. Working Example: Multi-Peak Fitting (XRD / Spectroscopy)

```python
import numpy as np
import matplotlib.pyplot as plt
from malthus_optimize import curve_fit

# Multi-peak model (Sum of two Gaussians + background)
def two_peaks(x, a1, mu1, sig1, a2, mu2, sig2, bg):
    g1 = a1 * np.exp(-0.5 * ((x - mu1) / sig1)**2)
    g2 = a2 * np.exp(-0.5 * ((x - mu2) / sig2)**2)
    return g1 + g2 + bg

# Synthesize noisy spectrometer data
x = np.linspace(10.0, 50.0, 200)
y_clean = two_peaks(x, 100.0, 25.0, 1.5, 75.0, 32.0, 2.0, 10.0)
y_obs = y_clean + np.random.normal(0, 3.0, size=len(x))

# Define physically plausible boundaries (NO sensitive p0 guess needed!)
bounds = (
    [10.0, 15.0, 0.5, 10.0, 25.0, 0.5, 0.0],   # Lower bounds
    [200.0, 30.0, 5.0, 200.0, 40.0, 5.0, 30.0]  # Upper bounds
)

popt, pcov = curve_fit(two_peaks, x, y_obs, bounds=bounds, seed=42)

# Extract standard errors from covariance diagonal
perr = np.sqrt(np.diag(pcov))

print("Fitted Peak 1 Center:", f"{popt[1]:.3f} ± {perr[1]:.3f}")
print("Fitted Peak 2 Center:", f"{popt[4]:.3f} ± {perr[4]:.3f}")
```

---

## 4. Covariance & Parameter Confidence

The returned `pcov` is the estimated covariance matrix of `popt`.
- **Standard Errors**: $\sigma_{\mathbf{p}} = \sqrt{\text{diag}(P_{\text{cov}})}$
- **Parameter Correlation**:
  $$R_{i, j} = \frac{P_{\text{cov}}[i, j]}{\sqrt{P_{\text{cov}}[i, i] \cdot P_{\text{cov}}[j, j]}}$$
This provides rigorous scientific uncertainty quantification identical to SciPy's output.
