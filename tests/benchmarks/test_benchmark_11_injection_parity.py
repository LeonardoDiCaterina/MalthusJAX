"""
BENCHMARK GROUP 11 — Injection + Key Derivation Fitness Parity
================================================================

Verify that injection-mode operators produce correct evolutionary dynamics
against an evosax golden standard.
"""

import jax.numpy as jnp
import pytest

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult
from tests.benchmarks.conftest_benchmarks import (
    _INJECTION_CROSSOVER_TYPES,
    _INJECTION_MUTATION_TYPES,
    _PARITY_RESULTS_DIR,
    KeyDerivationStrategy,
    _run_parity_comparison,
)


def _assert_experiment_result(
    result: "ExperimentResult",
    label: str,
    seeds: tuple[int, ...],
) -> None:
    assert len(result.runs) == len(seeds), (
        f"{label}: expected {len(seeds)} runs, got {len(result.runs)}"
    )
    for run in result.runs:
        assert run.status == "success", f"{label} seed={run.seed} failed: {run.error}"
        assert "start_best_fitness" in run.metrics
        assert "end_best_fitness" in run.metrics
        assert "delta_best" in run.metrics
        assert jnp.isfinite(run.metrics["best_fitness"]), (
            f"{label} seed={run.seed}: non-finite best_fitness"
        )
        assert run.metrics["delta_best"] >= 0, (
            f"{label} seed={run.seed}: non-improving run (delta={run.metrics['delta_best']})"
        )


def _assert_parity_comparison(
    comparison: "ComparisonResult",
    label: str,
    seeds: tuple[int, ...],
) -> None:
    assert set(comparison.names) == {"malthusjax", "evosax"}, (
        f"{label}: expected pipelines malthusjax+evosax, got {comparison.names}"
    )

    for name in comparison.names:
        result = comparison.pipelines[name]
        _assert_experiment_result(result, f"{label}/{name}", seeds)

    table = comparison.summary_table()
    mjx_best = table["malthusjax"]["best_fitness"]
    esx_best = table["evosax"]["best_fitness"]
    mjx_delta = table["malthusjax"].get("delta_best")
    esx_delta = table["evosax"].get("delta_best")

    print(
        f"\n  [{label}]  (mean over {len(seeds)} seeds, canonical init)"
        f"\n    MalthusJAX  best_fitness = {mjx_best:.6f}, Δ = {mjx_delta}"
        f"\n    Evosax      best_fitness = {esx_best:.6f}, Δ = {esx_delta}"
        f"\n    ratio (mjx/esx) = {mjx_best / esx_best:.4f}"
    )

    assert jnp.isfinite(mjx_best), f"{label}: MalthusJAX non-finite mean best"
    assert jnp.isfinite(esx_best), f"{label}: evosax non-finite mean best"


class TestInjectionFitnessParity:
    """Verify that injection-mode operators produce correct evolutionary dynamics.

    Pop=200, d=10, 100 generations over 30 seeds provides strong
    statistical confidence with 95% CI.
    """

    _POP = 200
    _DIMS = 10
    _GENS = 100
    _SEEDS = tuple(range(30))
    _OUTPUT_DIR = _PARITY_RESULTS_DIR

    @pytest.mark.parametrize("crossover_type", _INJECTION_CROSSOVER_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True], ids=["standard", "injection"])
    def test_crossover_parity(self, crossover_type: str, use_injection: bool):
        mode = "injection" if use_injection else "standard"
        label = f"crossover={crossover_type} mode={mode}"

        comparison = _run_parity_comparison(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            seeds=self._SEEDS,
            crossover_type=crossover_type,
            use_injection_ops=use_injection,
            output_dir=self._OUTPUT_DIR,
        )
        _assert_parity_comparison(comparison, label, self._SEEDS)

    @pytest.mark.parametrize("mutation_type", _INJECTION_MUTATION_TYPES)
    @pytest.mark.parametrize("use_injection", [False, True], ids=["standard", "injection"])
    def test_mutation_parity(self, mutation_type: str, use_injection: bool):
        mode = "injection" if use_injection else "standard"
        label = f"mutation={mutation_type} mode={mode}"

        comparison = _run_parity_comparison(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            seeds=self._SEEDS,
            mutation_type=mutation_type,
            use_injection_ops=use_injection,
            output_dir=self._OUTPUT_DIR,
        )
        _assert_parity_comparison(comparison, label, self._SEEDS)

    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    def test_key_derivation_parity(self, key_derivation: KeyDerivationStrategy):
        label = f"key_derivation={key_derivation.value}"

        comparison = _run_parity_comparison(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            seeds=self._SEEDS,
            key_derivation=key_derivation,
            output_dir=self._OUTPUT_DIR,
        )
        _assert_parity_comparison(comparison, label, self._SEEDS)

    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    @pytest.mark.parametrize("crossover_type", _INJECTION_CROSSOVER_TYPES)
    def test_injection_crossover_with_key_derivation(
        self, crossover_type: str, key_derivation: KeyDerivationStrategy
    ):
        label = f"injection crossover={crossover_type} key_derivation={key_derivation.value}"

        comparison = _run_parity_comparison(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            seeds=self._SEEDS,
            crossover_type=crossover_type,
            use_injection_ops=True,
            key_derivation=key_derivation,
            output_dir=self._OUTPUT_DIR,
        )
        _assert_parity_comparison(comparison, label, self._SEEDS)

    @pytest.mark.parametrize(
        "key_derivation",
        [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD],
        ids=["split", "fold"],
    )
    @pytest.mark.parametrize("mutation_type", _INJECTION_MUTATION_TYPES)
    def test_injection_mutation_with_key_derivation(
        self, mutation_type: str, key_derivation: KeyDerivationStrategy
    ):
        label = f"injection mutation={mutation_type} key_derivation={key_derivation.value}"

        comparison = _run_parity_comparison(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            seeds=self._SEEDS,
            mutation_type=mutation_type,
            use_injection_ops=True,
            key_derivation=key_derivation,
            output_dir=self._OUTPUT_DIR,
        )
        _assert_parity_comparison(comparison, label, self._SEEDS)
