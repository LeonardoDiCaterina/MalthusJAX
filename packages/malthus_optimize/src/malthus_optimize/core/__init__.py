"""Core data structures and utilities for malthus_optimize."""

from malthus_optimize.core.bounds import ParsedBounds, parse_bounds
from malthus_optimize.core.result import OptimizeResult

__all__ = ["OptimizeResult", "ParsedBounds", "parse_bounds"]
