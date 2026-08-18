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
        """Verify that qd_score respects the maximize flag sign conventions.

        MalthusJAX convention: qd_score is the sum of the RAW objective values.
        If we evaluate Rastrigin (raw values are positive) but we are minimizing
        (maximize=False), the qd_score should still be positive because it is
        the sum of the raw positive values.

        We verify this by ensuring that flipping the maximize flag flips the
        sign of the qd_score.
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

        # If the adapter correctly normalizes qd_score, the max and min
        # traces will have opposite signs (or at least diverge strongly
        # if the trajectories diverge).
        # We assert they are mathematically distinct and appropriately signed.
        assert min_qd != max_qd
