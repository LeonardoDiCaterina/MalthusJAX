"""Public SciPy-compatible API functions."""

from malthus_optimize.api.curve_fit import curve_fit
from malthus_optimize.api.differential_evolution import differential_evolution
from malthus_optimize.api.minimize import minimize

__all__ = ["differential_evolution", "minimize", "curve_fit"]
