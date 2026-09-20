import jax
import jax.numpy as jnp

# Choose the appropriate core evaluator
from malthusjax.core.fitness.composable.evaluators import (
    OptimizationEvaluator, SupervisedEvaluator, RLEvaluator
)
from malthusjax.core.fitness.composable.base import ScalarOutput, IdentityTransform
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

def create_custom_evaluator() -> OptimizationEvaluator:
    """Builds and returns a configured composable evaluator."""
    
    # 1. Environment
    # env = CustomEnv(...)
    env = None
    
    # 2. Transform (Optional, defaults to IdentityTransform)
    transform = IdentityTransform()
    
    # 3. Interpreter
    interpreter = IdentityInterpreter()
    
    # 4. Output Mode
    output = ScalarOutput(maximize=True)
    
    # 5. Composition
    return OptimizationEvaluator(
        env=env,
        transform=transform,
        interpreter=interpreter,
        output=output
    )
