import jax.numpy as jnp
import jax.random as jr


class QDAdapterTestSuiteMixin:
    """Mixin for QD-specific adapter tests.

    This mixin should be added to any BaseAdapterTestSuite that tests a QD adapter
    (like MAP-Elites) to ensure QD-specific metrics like qd_score and coverage
    are correctly routed and respect sign inversion conventions.
    """

    def test_qd_metrics_exist(self, small_adapter):
        """Verify that qd_score and coverage exist in the summary metrics."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        summary = result["summary"]

        assert "qd_score" in summary
        assert "coverage" in summary

        # Coverage should be a percentage [0, 100]
        assert 0.0 <= summary["coverage"] <= 100.0

    def test_qd_score_sign_consistency(self, small_adapter, small_adapter_max):
        """The score preserves higher-is-better archive space without sign inversion.

        This native QDax fixture uses the same scoring function for both flags,
        so both archives and their quality sums are identical.
        """
        key = jr.PRNGKey(42)
        min_result = small_adapter.run_once(key)
        max_result = small_adapter_max.run_once(key)

        min_qd = min_result["summary"]["qd_score"]
        max_qd = max_result["summary"]["qd_score"]

        assert jnp.isfinite(min_qd)
        assert jnp.isfinite(max_qd)
        assert min_qd != 0.0
        assert max_qd != 0.0

        assert min_qd == max_qd
        assert min_result["metric_metadata"]["metrics"]["qd_score"]["orientation"] == "higher_is_better"
        assert max_result["metric_metadata"]["quality_offset"] == 0.0
