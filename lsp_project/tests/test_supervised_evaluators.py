import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
from lsp.evaluator.supervised import SklearnMEPEvaluator
from lsp.evaluator.tensorneat_bridge import TensorNEATSupervisedProblem

def test_sklearn_mep_evaluator():
    config = LinearGenomeConfig(
        length=10, num_inputs=10, num_ops=5, max_arity=2
    )
    
    # We use a tiny synthetic regression dataset
    evaluator = SklearnMEPEvaluator(dataset="make_regression", n_samples=10, n_features=10)
    
    # Create dummy genome
    genome = LinearGenome(
        ops=jnp.zeros((10,), dtype=jnp.int32),
        args=jnp.zeros((10, 2), dtype=jnp.int32)
    )
    
    key = jax.random.PRNGKey(0)
    
    # Evaluate using the MalthusJAX MEP Evaluator
    fitness = evaluator.evaluate(genome, key)
    
    assert fitness is not None
    assert fitness.shape == ()

def test_tensorneat_bridge():
    # Instantiate the unified evaluator
    evaluator = SklearnMEPEvaluator(dataset="make_regression", n_samples=10, n_features=10)
    
    # Wrap it in TensorNEAT bridge
    problem = TensorNEATSupervisedProblem(evaluator)
    
    # Dummy state and forward pass function for TensorNEAT
    state = None
    params = None
    
    # A dummy forward pass that just returns the sum of inputs
    def dummy_act_func(state, params, inputs):
        return jnp.sum(inputs, axis=-1)
    
    key = jax.random.PRNGKey(0)
    
    # Evaluate via TensorNEAT bridge
    fitness = problem.evaluate(state, key, dummy_act_func, params)
    
    # Compute manually to assert equivalency
    X, y = evaluator.data
    outputs = jax.vmap(dummy_act_func, in_axes=(None, None, 0))(state, params, X)
    expected_fitness = -jnp.mean((jnp.squeeze(outputs) - y) ** 2)
    
    # The bridge should return exactly the expected negative MSE
    npt.assert_allclose(fitness, expected_fitness, rtol=1e-5)
