"""Composable fitness evaluation for MalthusJAX.

This package implements the three-axis evaluator decomposition:

    Problem  = TaskType × Environment × Interpreter
    Output   = ScalarOutput | QDOutput(DescriptorFn) | MOOutput
    Evaluator = Problem × Output

See ``docs/evaluator_design.md`` for the full design document.

Quick start::

    from malthusjax.core.fitness.composable import (
        SklearnEnv, MLPInterpreter, ScalarOutput, SupervisedEvaluator,
        BBOBEnv, IdentityInterpreter, OptimizationEvaluator,
    )

    # MLP on SKLearn
    evaluator = SupervisedEvaluator(
        env=SklearnEnv.create(dataset="breast_cancer"),
        interpreter=MLPInterpreter(input_dim=30, output_dim=1, hidden=(64, 64)),
        output=ScalarOutput(loss_fn="bce"),
    )

    # Standard GA on BBOB
    evaluator = OptimizationEvaluator(
        env=BBOBEnv.create(fn_name="sphere", num_dims=10),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
    )
"""

from malthusjax.core.fitness.composable.base import (
    BaseInterpreter,
    BaseEnvironment,
    BaseOptimizationEnvironment,
    BaseSupervisedEnvironment,
    BaseRLEnvironment,
    BaseDescriptorFn,
    ScalarOutput,
    QDOutput,
    MOOutput,
)
from malthusjax.core.fitness.composable.interpreters import (
    IdentityInterpreter,
    MLPInterpreter,
)
from malthusjax.core.fitness.composable.environments import (
    SklearnEnv,
    CustomDatasetEnv,
    BBOBEnv,
)
from malthusjax.core.fitness.composable.evaluators import (
    SupervisedEvaluator,
    OptimizationEvaluator,
)

__all__ = [
    # Base interfaces
    "BaseInterpreter",
    "BaseEnvironment",
    "BaseOptimizationEnvironment",
    "BaseSupervisedEnvironment",
    "BaseRLEnvironment",
    "BaseDescriptorFn",
    # Output modes
    "ScalarOutput",
    "QDOutput",
    "MOOutput",
    # Interpreters
    "IdentityInterpreter",
    "MLPInterpreter",
    # Environments
    "SklearnEnv",
    "CustomDatasetEnv",
    "BBOBEnv",
    # Evaluators
    "SupervisedEvaluator",
    "OptimizationEvaluator",
]
