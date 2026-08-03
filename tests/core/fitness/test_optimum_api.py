import jax
import pytest

from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


def test_base_evaluator_defaults():
    class DummyEval(BaseEvaluator):
        def evaluate(self, genome):
            return 1.0

    de = DummyEval(config=None, data=None)
    assert de.f_opt is None
    assert de.x_opt is None
    assert de.get_gap_to_optimum(1.0) is None


def test_bbob_evaluator_optimum_and_gap():
    cfg = BBOBConfig(fn_name="sphere", num_dims=3, seed=0, maximize=False)
    ev = BBOBEvaluator.create(cfg)

    # f_opt and x_opt should be provided by the underlying evosax problem
    assert ev.f_opt is not None
    assert ev.x_opt is not None

    # Evaluate the optimum via the evosax problem directly and check gap is ~0
    opt_x = ev.x_opt
    fval = ev.evosax_problem.eval(jax.random.PRNGKey(0), opt_x[None, :], ev.problem_state)[0][0]
    gap = ev.get_gap_to_optimum(fval)
    assert gap is not None
    assert float(gap) == pytest.approx(0.0, abs=1e-6)
