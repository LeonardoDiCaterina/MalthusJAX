"""P3 golden archives: occupancy, orientation, dispersion and shared contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.benchmarking.empirical import qd_comparison_status
from malthusjax.composer.adapters.metrics import archive_metric_metadata, archive_metric_specs
from malthusjax.core.fitness.qd.metrics import archive_metrics


@pytest.mark.parametrize("maximize", [False, True])
def test_same_archive_native_and_upstream_output_semantics(maximize):
    # Repertoire storage is always higher-is-better.
    quality = jnp.array([2.0, 4.0, -jnp.inf]) * (1 if maximize else -1)
    quality = quality.at[2].set(-jnp.inf)
    metrics = jax.jit(archive_metrics)(quality)
    native = {
        "best_fitness": -metrics["max_fitness"],
        "mean_fitness": -metrics["mean_fitness"],
        "std_fitness": metrics["std_fitness"],
        "qd_score": metrics["qd_score"],
    }
    for key in ("best_fitness", "mean_fitness"):
        native[key] *= -1 if maximize else 1
    upstream = {
        m.name: metrics[m.source] * (-1 if m.is_objective_value and not maximize else 1)
        for m in archive_metric_specs()
    }
    for name, value in native.items():
        np.testing.assert_allclose(value, upstream[name])
    assert float(native["std_fitness"]) == 1.0
    assert float(native["mean_fitness"]) == 3.0
    assert float(native["best_fitness"]) == (4.0 if maximize else 2.0)
    assert float(native["qd_score"]) == (6.0 if maximize else -6.0)
    assert float(metrics["coverage"]) == pytest.approx(200 / 3, abs=1e-5)


def test_empty_archive_is_explicitly_unavailable():
    result = archive_metrics(jnp.full((4,), -jnp.inf))
    assert result["coverage"] == 0
    assert result["occupied_cells"] == 0
    assert all(
        np.isnan(result[k]) for k in ("max_fitness", "mean_fitness", "std_fitness", "qd_score")
    )


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf])
def test_invalid_cells_are_not_silently_treated_as_empty(bad):
    result = archive_metrics(jnp.array([2.0, bad, -jnp.inf]))
    assert np.isnan(result["coverage"])
    assert np.isnan(result["qd_score"])


def test_additional_empty_cells_cannot_change_occupied_statistics():
    left = archive_metrics(jnp.array([-2.0, -4.0]))
    right = archive_metrics(jnp.array([-2.0, -4.0, -jnp.inf, -jnp.inf]))
    for name in ("max_fitness", "mean_fitness", "std_fitness", "qd_score"):
        np.testing.assert_allclose(left[name], right[name])


def test_comparability_requires_matching_versioned_contracts():
    row = {"metric_metadata": archive_metric_metadata(False)}
    assert qd_comparison_status([row], [row])["status"] == "comparable"
    assert qd_comparison_status([row], [{}])["status"] == "unavailable"
    assert (
        qd_comparison_status([row], [{"metric_metadata": archive_metric_metadata(True)}])["status"]
        == "unavailable"
    )
