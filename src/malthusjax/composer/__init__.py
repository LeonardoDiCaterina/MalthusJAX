"""Composer for building evolutionary experiments.

The Composer provides a product-first API for quickly running experiments
with sensible defaults and declarative configuration.
"""

from .composer import Composer
from .decorators import (
    register_backend,
    register_crossover,
    register_emitter,
    register_engine,
    register_fitness,
    register_genome,
    register_mutation,
    register_selection,
)
from .engine_catalog import EngineRegistry
from .engine_factory_v2 import EngineFactory
from .engine_protocol import Engine, RunOutput, StepMetrics
from .evosax_adapter import EvosaxEngineAdapter, build_evosax_engine, list_strategies
from .experiment_config import (
    BackendConfig,
    EvosaxBackendConfig,
    ExecutionConfig,
    ExperimentConfig,
    GenericBackendConfig,
    LoggingConfig,
    MalthusJAXBackendConfig,
    OutputConfig,
    PopulationConfig,
    QdaxBackendConfig,
    StubBackendConfig,
    TensorneatBackendConfig,
)
from .mo_factory import MOEngineAdapter, build_mo_engine

__all__ = [
    "Composer",
    "Engine",
    "EngineFactory",
    "EngineRegistry",
    "RunOutput",
    "StepMetrics",
    "EvosaxEngineAdapter",
    "build_evosax_engine",
    "list_strategies",
    "register_backend",
    "register_selection",
    "register_mutation",
    "register_crossover",
    "register_emitter",
    "register_fitness",
    "register_engine",
    "register_genome",
    "MOEngineAdapter",
    "build_mo_engine",
    "ExperimentConfig",
    "PopulationConfig",
    "ExecutionConfig",
    "LoggingConfig",
    "OutputConfig",
    "BackendConfig",
    "MalthusJAXBackendConfig",
    "EvosaxBackendConfig",
    "QdaxBackendConfig",
    "TensorneatBackendConfig",
    "StubBackendConfig",
    "GenericBackendConfig",
]


# Ensure built-in backends are registered
import malthusjax.composer.backends  # noqa: F401

# Auto-discover plugins on import
from .discovery import discover_plugins

discover_plugins()
