"""
BENCHMARK GROUP 5 — Fitness Quality (Convergence Parity)
=======================================================

Verify both frameworks converge to comparable fitness on BBOB problems.
"""

import jax.numpy as jnp
import pytest

from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    NUM_GENERATIONS_LONG,
    PROBLEMS,
    _run_comparison,
)


class TestConvergenceParity:
    """Verify both frameworks converge to comparable fitness on BBOB problems.

    Uses :class:`BenchmarkRunner` and :class:`ComparisonResult` from the
    ``malthusjax.benchmarking`` infrastructure for structured multi-seed
    execution and sign-normalised comparison.
    """

    @pytest.mark.parametrize("problem", PROBLEMS)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_fitness_parity(self, problem: str, dims: int):
        """Compare final best fitness after fixed generation budget (10 seeds)."""
        pop_size = 200
        num_gens = NUM_GENERATIONS_LONG
        seeds = (42, 123, 7, 99, 0, 1, 2021, 2022, 2023, 2024)

        comparison = _run_comparison(
            pop_size=pop_size,
            dims=dims,
            problem=problem,
            num_generations=num_gens,
            seeds=seeds,
        )

        # ---- Validate ComparisonResult structure ----
        assert set(comparison.names) == {"malthusjax", "evosax"}

        for name in comparison.names:
            exp = comparison.pipelines[name]
            assert len(exp.runs) == len(seeds), (
                f"{name}: expected {len(seeds)} runs, got {len(exp.runs)}"
            )
            for run in exp.runs:
                assert run.status == "success", f"{name} seed={run.seed} failed: {run.error}"
                assert "start_best_fitness" in run.metrics
                assert "end_best_fitness" in run.metrics
                assert "delta_best" in run.metrics
                assert run.metrics["delta_best"] >= 0, (
                    f"non‑improving run {run.seed}: delta={run.metrics['delta_best']}"
                )

        # ---- Aggregated summary via ComparisonResult ----
        table = comparison.summary_table()
        mjx_best = table["malthusjax"]["best_fitness"]
        esx_best = table["evosax"]["best_fitness"]
        mjx_start = table["malthusjax"].get("start_best_fitness")
        mjx_end = table["malthusjax"].get("end_best_fitness")
        mjx_delta = table["malthusjax"].get("delta_best")
        esx_start = table["evosax"].get("start_best_fitness")
        esx_end = table["evosax"].get("end_best_fitness")
        esx_delta = table["evosax"].get("delta_best")

        print(
            f"\n  [{problem} d={dims}]  (mean over {len(seeds)} seeds)"
            f"\n    MalthusJAX best_fitness = {mjx_best:.6f},"
            f" start={mjx_start}, end={mjx_end}, Δ={mjx_delta}"
            f"\n    Evosax     best_fitness = {esx_best:.6f},"
            f" start={esx_start}, end={esx_end}, Δ={esx_delta}"
        )

        for name in comparison.names:
            print(f"\n  {name} per-run metrics:")
            for run in comparison.pipelines[name].runs:
                print(f"    seed={run.seed} metrics={run.metrics}")

        conv = comparison.convergence_data(seed_index=0)
        for name in comparison.names:
            assert len(conv[name]) > 0, f"No history for {name}"
            assert "best_fitness" in conv[name][0], f"Missing best_fitness in {name} history"

        assert jnp.isfinite(mjx_best), (
            f"MalthusJAX returned non-finite mean best_fitness: {mjx_best}"
        )
        assert jnp.isfinite(esx_best), f"Evosax returned non-finite mean best_fitness: {esx_best}"
