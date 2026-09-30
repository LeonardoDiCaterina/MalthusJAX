"""Tests for Engine protocol and runtime checkability."""

import chex

from malthusjax.benchmarking import StubEngine
from malthusjax.composer.engine_protocol import Engine, RunOutput, StepMetrics


class CustomValidEngine:
    def run_once(self, key: chex.Array, **kwargs) -> RunOutput:
        return {
            "history": [{"generation": 1, "best_fitness": 0.0}],
            "summary": {"best_fitness": 0.0},
            "timings": {"warmup": 0.01, "execution": 0.05, "total": 0.06},
        }


class InvalidEngineMissingMethod:
    def execute(self, key: chex.Array):
        return {}


def test_stub_engine_satisfies_protocol():
    """StubEngine satisfies the runtime-checkable Engine protocol."""
    engine = StubEngine(generations=10)
    assert isinstance(engine, Engine)


def test_custom_engine_satisfies_protocol():
    """Custom user class with run_once() satisfies the Engine protocol."""
    engine = CustomValidEngine()
    assert isinstance(engine, Engine)


def test_non_engine_fails_check():
    """Class missing run_once() fails isinstance(obj, Engine)."""
    invalid = InvalidEngineMissingMethod()
    assert not isinstance(invalid, Engine)

    non_obj = 42
    assert not isinstance(non_obj, Engine)


def test_run_output_structure():
    """RunOutput TypedDict holds expected keys."""
    metrics: StepMetrics = {
        "best_fitness": 1.23,
        "mean_fitness": 2.34,
        "std_fitness": 0.5,
        "generation": 1,
    }
    output: RunOutput = {
        "history": [metrics],
        "summary": {"best_fitness": 1.23},
        "timings": {"total": 0.1},
    }
    assert output["history"][0]["best_fitness"] == 1.23
    assert output["summary"]["best_fitness"] == 1.23
