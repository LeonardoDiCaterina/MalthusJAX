from malthusjax.composer.composer import Composer
from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.environments import GymnaxEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter


def test_composable_evosax_adapter_dimension_extraction():
    composer = Composer()

    # 1. Build an RLEvaluator manually
    env = GymnaxEnv.create(env_name="CartPole-v1")
    interpreter = MLPInterpreter(
        input_dim=env.obs_dim,
        output_dim=env.action_dim,
        hidden=(32,)
    )
    output = ScalarOutput(maximize=True)
    evaluator = RLEvaluator(env=env, interpreter=interpreter, output=output, max_steps=100)

    # 2. Use Composer to run EvoSAX on the RLEvaluator natively using composable backend!
    result = composer.quick_run(
        backend="composable_evosax",
        evosax_strategy="SimpleGA",
        fitness=evaluator,
        pop_size=20,
        generations=2,
        maximize=True,
    )

    assert hasattr(result, "runs")
    assert len(result.runs) > 0
    assert "best_fitness" in result.runs[0].metrics
