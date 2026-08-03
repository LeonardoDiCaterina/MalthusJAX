"""Benchmarking helpers for MalthusJAX.

This package contains dataclasses and IO helpers used by the BenchmarkRunner.
"""

from .io import ensure_seed_folder, read_summary_json, write_histories_csv, write_summary_json
from .results import ComparisonResult, ExperimentResult, RunResult
from .runner import BenchmarkRunner, Engine, StubEngine

__all__ = [
    "RunResult",
    "ExperimentResult",
    "ComparisonResult",
    "write_summary_json",
    "read_summary_json",
    "write_histories_csv",
    "ensure_seed_folder",
    "BenchmarkRunner",
    "Engine",
    "StubEngine",
]
