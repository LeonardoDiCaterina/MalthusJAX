import abc

import jax.numpy as jnp
import jax.random as jr
import pytest


class BaseAdapterTestSuite(abc.ABC):
    """Base test suite for MalthusJAX engine adapters.

    Any adapter wrapping an external framework (e.g. Evosax, QDax, TensorNEAT)
    should inherit from this class and implement `make_adapter`. This ensures
    the adapter perfectly adheres to the MalthusJAX Engine protocol.
    """

    @abc.abstractmethod
    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        """Constructs and returns a minimal, fast-running Engine adapter instance.

        Args:
            maximize: Whether the problem is a maximization problem.
            eval_mode: The evaluation mode (e.g. "native" or "malthusjax").
            seed: Random seed for initialization.

        Returns:
            An instance of a class that conforms to the MalthusJAX Engine protocol.
        """
        pass

    @pytest.fixture
    def small_adapter(self):
        """Fixture that returns a standard minimisation adapter."""
        return self.make_adapter(maximize=False)

    @pytest.fixture
    def small_adapter_max(self):
        """Fixture that returns a standard maximisation adapter."""
        return self.make_adapter(maximize=True)

    @pytest.fixture
    def small_adapter_mjx(self):
        """Fixture that returns a MALTHUSJAX eval mode adapter."""
        return self.make_adapter(eval_mode="malthusjax")

    # ---------------------------------------------------------------------------
    # Protocol Conformance (run_once)
    # ---------------------------------------------------------------------------

    def test_result_keys(self, small_adapter):
        """Verify the output of run_once contains exactly the required keys."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        assert set(result.keys()) == {"history", "summary", "timings"}

    def test_history_format(self, small_adapter):
        """Verify the history is a list of dictionaries with standard keys."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        history = result["history"]

        assert isinstance(history, list)
        assert len(history) > 0

        required_keys = {"generation", "best_fitness"}
        for entry in history:
            assert required_keys.issubset(entry.keys())

    def test_history_generations_sequential(self, small_adapter):
        """Verify generations are emitted in sequential order starting at 1."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        gens = [h["generation"] for h in result["history"]]
        expected_gens = list(range(1, len(gens) + 1))
        assert gens == expected_gens

    def test_summary_format(self, small_adapter):
        """Verify the summary dictionary contains required end-of-run metrics."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        summary = result["summary"]

        assert "best_fitness" in summary
        assert "final_generation" in summary
        assert "total_evaluations" in summary

        history = result["history"]
        assert summary["final_generation"] == history[-1]["generation"]
        assert summary["total_evaluations"] > 0

    def test_timings_format(self, small_adapter):
        """Verify timings are present and strictly positive."""
        result = small_adapter.run_once(jr.PRNGKey(0))
        timings = result["timings"]

        assert "warmup" in timings
        assert "execution" in timings
        assert "total" in timings
        assert timings["warmup"] >= 0
        assert timings["execution"] > 0
        assert timings["total"] > 0

    def test_fitness_values_are_finite(self, small_adapter):
        """Verify that fitness values do not contain NaNs or Infs."""
        result = small_adapter.run_once(jr.PRNGKey(42))

        for entry in result["history"]:
            assert jnp.isfinite(entry["best_fitness"]), f"gen {entry['generation']}"
        assert jnp.isfinite(result["summary"]["best_fitness"])

    # ---------------------------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------------------------

    def test_same_key_same_result(self, small_adapter):
        """Verify identical PRNG keys yield identical execution traces."""
        key = jr.PRNGKey(999)
        r1 = small_adapter.run_once(key)
        r2 = small_adapter.run_once(key)

        assert r1["summary"]["best_fitness"] == r2["summary"]["best_fitness"]
        assert len(r1["history"]) == len(r2["history"])
        for h1, h2 in zip(r1["history"], r2["history"]):
            assert h1["best_fitness"] == h2["best_fitness"]

    def test_different_keys_different_results(self, small_adapter):
        """Verify different PRNG keys yield distinct trajectories."""
        r1 = small_adapter.run_once(jr.PRNGKey(0))
        r2 = small_adapter.run_once(jr.PRNGKey(12345))

        # Overwhelmingly unlikely to match on different keys for continuous/complex spaces
        assert r1["summary"]["best_fitness"] != r2["summary"]["best_fitness"]

    # ---------------------------------------------------------------------------
    # Maximisation Convention Mapping
    # ---------------------------------------------------------------------------

    def test_maximize_flag_changes_outcome(self, small_adapter, small_adapter_max):
        """Verify that toggling maximize changes the fitness optimization trajectory."""
        key = jr.PRNGKey(42)
        min_bf = small_adapter.run_once(key)["summary"]["best_fitness"]
        max_bf = small_adapter_max.run_once(key)["summary"]["best_fitness"]
        assert min_bf != max_bf

    def test_maximize_history_changes(self, small_adapter, small_adapter_max):
        """Verify that toggling maximize changes the intermediate history."""
        key = jr.PRNGKey(42)
        min_hist = small_adapter.run_once(key)["history"]
        max_hist = small_adapter_max.run_once(key)["history"]

        # at least one generation should differ
        assert any(
            hmin["best_fitness"] != hmax["best_fitness"] for hmin, hmax in zip(min_hist, max_hist)
        )

    # ---------------------------------------------------------------------------
    # EvalMode bridging
    # ---------------------------------------------------------------------------

    def test_malthusjax_mode_is_supported(self, small_adapter_mjx):
        """Verify that the adapter can run properly with a generic MalthusJAX evaluator."""
        # Note: If the adapter does not support MALTHUSJAX mode yet, the subclass
        # should mark this test as xfail or skip it. If small_adapter_mjx constructs
        # properly, it should be able to run.
        if small_adapter_mjx is None:
            pytest.skip("EvalMode.MALTHUSJAX not implemented for this adapter.")

        result = small_adapter_mjx.run_once(jr.PRNGKey(0))
        assert set(result.keys()) == {"history", "summary", "timings"}
        assert len(result["history"]) > 0
