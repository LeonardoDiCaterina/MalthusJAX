import jax
import pytest

pytest.importorskip("evosax")

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import BBOBEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter


def test_evosax_adapter_includes_gap():
    pop_size = 20
    generations = 5
    # Build a BBOB evaluator and an evosax adapter, run a short experiment,
    # and assert the returned summary contains gap_to_optimum when available.
    ev = OptimizationEvaluator(
        env=BBOBEnv.create(fn_name="sphere", num_dims=5, seed=0),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )

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
