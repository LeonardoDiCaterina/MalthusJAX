import jax
import jax.numpy as jnp

class TensorNEATSupervisedProblem:
    """
    A bridge wrapper that allows TensorNEAT to evaluate genomes against 
    the unified SklearnEvaluator or EquationEvaluator. 
    
    This ensures that TensorNEAT uses the exact same JAX data matrices 
    and loss functions as the MalthusJAX symbolic representations.
    """
    def __init__(self, evaluator):
        """
        Args:
            evaluator: An instantiated BaseEvaluator (e.g. SklearnMEPEvaluator)
                       that holds the batched data in self.data = (X, y).
        """
        self.evaluator = evaluator
        
    def setup(self, state=None):
        return state

    def evaluate(self, state, randkey, act_func, params):
        # Extract the cached dataset from the unified evaluator
        X, y = self.evaluator.data
        
        # act_func is the TensorNEAT forward pass. It typically expects
        # (state, params, inputs). We vmap it over the batch dimension of X.
        outputs = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, X)
        
        # Ensure predictions match the target shape (typically single-output regression)
        outputs = jnp.squeeze(outputs)
        
        # Compute Mean Squared Error (MSE)
        mse = jnp.mean((outputs - y) ** 2)
        
        # MalthusJAX's TensorNEAT adapter natively handles the negation 
        # based on the maximize flag. We must return the raw objective value.
        return mse
