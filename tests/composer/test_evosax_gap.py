import jax
import pytest

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.composer.evosax_adapter import build_evosax_engine


@pytest.mark.parametrize("pop_size,generations", [(8, 2), (16, 3)])
def test_evosax_adapter_includes_gap(pop_size, generations):
    # Build a BBOB evaluator and an evosax adapter, run a short experiment,
    # and assert the returned summary contains gap_to_optimum when available.
    ev = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=5, seed=0, maximize=False))

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=ev,
        pop_size=pop_size,
        generations=generations,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=0,
    )

    res = adapter.run_once(jax.random.PRNGKey(0), compile=False)
    assert "summary" in res
    summary = res["summary"]
    # If evaluator exposes an optimum, gap_to_optimum should be present and numeric
    assert "gap_to_optimum" in summary
    assert isinstance(summary["gap_to_optimum"], float)
