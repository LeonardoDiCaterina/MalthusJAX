"""malthus_optimize: SciPy-compatible global optimization powered by JAX and MalthusJAX Composer.

Provides high-throughput, GPU/TPU-accelerated drop-in replacements for:
- scipy.optimize.differential_evolution
- scipy.optimize.minimize
- scipy.optimize.curve_fit
- scipy.optimize.OptimizeResult
"""

from __future__ import annotations

from malthus_optimize.api.curve_fit import curve_fit
from malthus_optimize.api.differential_evolution import differential_evolution
from malthus_optimize.api.minimize import minimize
from malthus_optimize.core.bounds import ParsedBounds, parse_bounds
from malthus_optimize.core.result import OptimizeResult

__version__ = "0.1.0"

__all__ = [
    "differential_evolution",
    "minimize",
    "curve_fit",
    "OptimizeResult",
    "ParsedBounds",
    "parse_bounds",
]
