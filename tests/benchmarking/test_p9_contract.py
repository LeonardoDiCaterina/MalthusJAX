import json
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.benchmarking.results import RunResult, standard_json
from malthusjax.benchmarking.runner import BenchmarkRunner, StubEngine


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_objective_never_success(value):
    class Engine:
        def run_once(self, key):
            return {"summary": {"best_fitness": value}}

    result = BenchmarkRunner(Engine(), write_artifacts=False).run([0]).runs[0]
    assert result.status != "success"


def test_timeout_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="timeout"):
        BenchmarkRunner(StubEngine(), write_artifacts=False).run([0], timeout_seconds=1)


def test_standard_json_null_and_immutable_snapshots(tmp_path):
    payload = RunResult(0, "error", {"bad": np.nan}).to_json()
    assert json.loads(payload)["metrics"]["bad"] is None
    runner = BenchmarkRunner(StubEngine(), output_dir=tmp_path)
    runner.run([0])
    first = list(tmp_path.glob("manifest-*.json"))[0]
    original = first.read_bytes()
    runner.run([1])
    assert first.read_bytes() == original
    assert len(list(tmp_path.glob("manifest-*.json"))) == 2
    assert json.loads(original)["experiment"]["runs"][0]["seed"] == 0


def test_serialized_legacy_nan_cannot_claim_success():
    record = RunResult(0, "success", {"best_fitness": float("nan")})
    restored = RunResult.from_dict(json.loads(record.to_json()))
    assert restored.status == "invalid"
    assert restored.unavailable_metrics["best_fitness"] == "nan"


def test_standard_json_only_converts_numeric_arrays():
    unknown = MagicMock()
    payload = standard_json(
        {"unknown": unknown, "numpy": np.array([1, np.nan]), "jax": jnp.array([2])}
    )
    unknown.tolist.assert_not_called()
    assert payload["unknown"]["unserializable_type"] == "unittest.mock.MagicMock"
    assert payload["numpy"] == [1.0, None]
    assert payload["jax"] == [2]
    json.dumps(payload, allow_nan=False)
