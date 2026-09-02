import jax
import jax.numpy as jnp
import pytest

from lsp.genome.neural_linear import NeuralPrefixGenomeConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluatorConfig, NeuralPrefixEvaluator


def test_global_softmax_forward_pass():
    """Verify that the global softmax routing evaluates without shape errors."""
    key = jax.random.PRNGKey(42)
    
    config = NeuralPrefixGenomeConfig(
        num_inputs=2, length=5, num_ops=4, max_arity=2
    )
    
    # 4 samples, 2 features
    X = jax.random.normal(key, (4, 2))
    # 4 targets
    y = jax.random.normal(key, (4,))
    
    eval_config = NeuralPrefixEvaluatorConfig(
        num_inputs=2, length=5, loss_function="mse",
        mep_output_strategy="global_softmax", attention_temperature=0.5
    )
    evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, y))
    
    # Init pop
    pop = config.init_population(key, size=1)
    genome = jax.tree.map(lambda x: x[0], pop.genes)
    
    loss = evaluator.evaluate(genome, key)
    
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert not jnp.isnan(loss), "Loss is NaN"


def test_global_softmax_gradients():
    """Verify that gradients propagate to continuous weights through the softmax."""
    key = jax.random.PRNGKey(42)
    
    config = NeuralPrefixGenomeConfig(
        num_inputs=2, length=5, num_ops=4, max_arity=2
    )
    
    X = jax.random.normal(key, (4, 2))
    y = jax.random.normal(key, (4,))
    
    eval_config = NeuralPrefixEvaluatorConfig(
        num_inputs=2, length=5, loss_function="mse",
        mep_output_strategy="global_softmax", attention_temperature=0.5
    )
    evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, y))
    
    pop = config.init_population(key, size=1)
    genome = jax.tree.map(lambda x: x[0], pop.genes)
    
    def loss_fn(weights, biases):
        new_genome = genome.replace(weights=weights, biases=biases)
        return evaluator.evaluate(new_genome, key)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grads_w, grads_b = grad_fn(genome.weights, genome.biases)
    
    # Assert gradients are not zero
    assert jnp.sum(jnp.abs(grads_w)) > 0.0, "Gradients for weights should be non-zero"
    assert jnp.sum(jnp.abs(grads_b)) > 0.0, "Gradients for biases should be non-zero"


def test_global_softmax_bce_loss():
    """Verify global softmax routing works with BCE loss."""
    key = jax.random.PRNGKey(42)
    
    config = NeuralPrefixGenomeConfig(
        num_inputs=2, length=5, num_ops=4, max_arity=2
    )
    
    X = jax.random.normal(key, (4, 2))
    # Binary targets
    y = jax.random.bernoulli(key, 0.5, shape=(4,)).astype(jnp.float32)
    
    eval_config = NeuralPrefixEvaluatorConfig(
        num_inputs=2, length=5, loss_function="bce",
        mep_output_strategy="global_softmax", attention_temperature=0.5
    )
    evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, y))
    
    pop = config.init_population(key, size=1)
    genome = jax.tree.map(lambda x: x[0], pop.genes)
    
    loss = evaluator.evaluate(genome, key)
    
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert not jnp.isnan(loss), "Loss is NaN"
