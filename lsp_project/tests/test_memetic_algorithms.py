import jax
import jax.numpy as jnp
import optax
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig


def test_lsmf_gradient_flow():
    """Verify that continuous weights change while discrete topology remains untouched."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # 1. Setup config and mock data
    gc = NeuralPrefixGenomeConfig(length=5, num_inputs=2, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=2, length=5, loss_function="mse")

    X = jax.random.normal(k1, (10, 2))
    y = jax.random.normal(k2, (10,))

    base_eval = NeuralPrefixEvaluator(config=ec, data=(X, y))

    # 2. Setup LSMF
    optimizer = optax.adam(learning_rate=0.1)
    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_eval, optimizer=optimizer, epochs=5)

    # 3. Create a population of size 2
    population = gc.init_population(k3, 2)

    original_weights = population.genes.weights
    original_biases = population.genes.biases
    original_ops = population.genes.ops
    original_args = population.genes.args

    # 4. Run evaluate_population
    updated_population = lsmf.evaluate_population(population, key=k4)

    # 5. Assertions
    # Continuous parameters should have changed
    assert not jnp.allclose(original_weights, updated_population.genes.weights)
    assert not jnp.allclose(original_biases, updated_population.genes.biases)

    # Discrete parameters should remain exactly the same
    assert jnp.array_equal(original_ops, updated_population.genes.ops)
    assert jnp.array_equal(original_args, updated_population.genes.args)

def test_dmep_gradient_routing():
    """Verify that in dMEP dynamic routing, only the active subgraph gets gradients."""
    # This is a bit advanced to test automatically because we need to know which node
    # was the argmin and assert that ONLY its ancestral weights changed.
    # We will approximate this by verifying that NOT ALL weights change when dynamic routing is used,
    # or just checking the core mechanics of SGD.

    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Use a large length so it's highly likely that some nodes are completely disconnected
    # from the winning node, and therefore should receive 0 gradient.
    gc = NeuralPrefixGenomeConfig(length=20, num_inputs=2, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=2, length=20, loss_function="mse")

    X = jax.random.normal(k1, (10, 2))
    # Shift y by 100.0 so that a dead ReLU (output 0) has a terrible loss (~10000).
    # This prevents the dynamic routing from picking a dead node with 0 gradient!
    y = jax.random.normal(k2, (10,)) + 100.0

    base_eval = NeuralPrefixEvaluator(config=ec, data=(X, y))
    optimizer = optax.adam(learning_rate=0.1)
    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_eval, optimizer=optimizer, epochs=5)

    population = gc.init_population(k3, 1)
    updated_population = lsmf.evaluate_population(population, key=k4)

    # Some weights should change (the active ones), but because adam maintains state,
    # actually adam will update all parameters by their learning rate if gradient is 0
    # (wait, adam update with 0 grad and 0 m/v might just be 0, unless there's weight decay).
    # Since we have no weight decay, nodes with 0 gradient should have exactly 0 change.

    diff = jnp.abs(population.genes.weights - updated_population.genes.weights)

    # Assert that at least some weights changed
    assert jnp.max(diff) > 0.0

    # Assert that some weights did NOT change (because they weren't in the active graph)
    # This might flake if by sheer luck every node is connected, but with length 20 and arity 2 it's very likely some are dead.
    assert jnp.min(diff) == 0.0
