"""
Core module for MalthusJAX.

This module contains the foundational Level 1 components:
- Base abstractions with JAX-native design
- Genome representations with automatic vectorization
- Fitness evaluation functions with symbiotic support
- Modern population management
"""

from . import base, diagnostics, fitness, genome, logger

# Expose new base classes
from .base import BaseGenome, BasePopulation, DistanceMetric
from .diagnostics import (
    format_crash_banner,
    get_environment_diagnostics,
    install_crash_handler,
    print_environment_diagnostics,
    stabilize_runtime_environment,
    uninstall_crash_handler,
)
from .logger import (
    StepLoggingConfig,
    configure_logging,
    get_logger,
    set_log_level,
)
from .random import PRNGImpl, create_key

__all__ = [
    "base",
    "genome",
    "fitness",
    "logger",
    "diagnostics",
    "BaseGenome",
    "BasePopulation",
    "DistanceMetric",
    "PRNGImpl",
    "create_key",
    "get_logger",
    "set_log_level",
    "configure_logging",
    "StepLoggingConfig",
    "stabilize_runtime_environment",
    "install_crash_handler",
    "uninstall_crash_handler",
    "format_crash_banner",
    "get_environment_diagnostics",
    "print_environment_diagnostics",
]
