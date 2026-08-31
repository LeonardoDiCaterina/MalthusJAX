import jax
import jax.numpy as jnp
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.genome.neural_linear import NeuralPrefixGenome, NeuralPrefixGenomeConfig


def test_stochastic_batching():
    """Verify that evaluators correctly draw random mini-batches without breaking vectorization."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Create dataset of 100 samples
    X = jax.random.normal(k1, (100, 5))
    y = jax.random.normal(k2, (100,))

    # Configure evaluator with batch_size 8
    gc = NeuralPrefixGenomeConfig(length=20, num_inputs=5, num_ops=5, max_arity=2)
    ec = NeuralPrefixEvaluatorConfig(num_inputs=5, length=20, batch_size=8)

    evaluator = NeuralPrefixEvaluator(config=ec, data=(X, y))
    genome = NeuralPrefixGenome.random_init(k3, gc)

    # 1. Test without RNG (Should evaluate full batch)
    loss_full = evaluator.evaluate(genome, None)
    assert loss_full.shape == ()

    # 2. Test with RNG (Should slice to 8 samples)
    loss_batch = evaluator.evaluate(genome, k4)
    assert loss_batch.shape == ()

    # Since batch and full are different samples, loss should differ
    assert not jnp.isclose(loss_full, loss_batch)

def test_dmep_output_strategies():
    """Verify dMEP can properly route dynamic, out_nodes, and linear_readout."""
    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (10, 5))

    def evaluate_strategy(strategy, loss, y_shape, num_outputs):
        gc = NeuralPrefixGenomeConfig(
            length=10,
            num_inputs=5,
            num_ops=5,
            max_arity=2,
            num_outputs=num_outputs,
            mep_output_strategy=strategy
        )
        ec = NeuralPrefixEvaluatorConfig(num_inputs=5, length=10, loss_function=loss)

        y = jax.random.normal(k2, y_shape) if loss == "mse" else jax.random.randint(k2, y_shape, 0, num_outputs)
        evaluator = NeuralPrefixEvaluator(config=ec, data=(X, y))
        genome = NeuralPrefixGenome.random_init(k3, gc)

        return evaluator.evaluate(genome, None)

    # 1. Dynamic MSE
    loss_dyn = evaluate_strategy("dynamic", "mse", (10,), 1)
    assert loss_dyn.shape == ()

    # 2. Out Nodes CCE
    loss_out = evaluate_strategy("out_nodes", "cce", (10,), 3)
    assert loss_out.shape == ()

    # 3. Linear Readout CCE
    loss_readout = evaluate_strategy("linear_readout", "cce", (10,), 5)
    assert loss_readout.shape == ()

def test_dynamic_loss_functions():
    """Verify loss functions compute scalars."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (10, 5))
    y_mse = jax.random.normal(k2, (10, 3))
    y_cce = jax.random.randint(k2, (10,), 0, 3)

    gc = NeuralPrefixGenomeConfig(length=10, num_inputs=5, num_ops=5, max_arity=2, num_outputs=3, mep_output_strategy="linear_readout")
    genome = NeuralPrefixGenome.random_init(k3, gc)

    # Test MSE
    ec_mse = NeuralPrefixEvaluatorConfig(num_inputs=5, length=10, loss_function="mse")
    loss_mse = NeuralPrefixEvaluator(config=ec_mse, data=(X, y_mse)).evaluate(genome)
    assert not jnp.isnan(loss_mse)

    # Test CCE
    ec_cce = NeuralPrefixEvaluatorConfig(num_inputs=5, length=10, loss_function="cce")
    loss_cce = NeuralPrefixEvaluator(config=ec_cce, data=(X, y_cce)).evaluate(genome)
    assert not jnp.isnan(loss_cce)

import optax
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator


def test_lsmf_evaluator_updates():
    """Verify LSMFEvaluator performs Lamarckian weight updates."""
    key = jax.random.PRNGKey(111)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    X = jax.random.normal(k1, (20, 3))
    y = jax.random.normal(k2, (20,))

    gc = NeuralPrefixGenomeConfig(length=10, num_inputs=3, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=3, length=10, loss_function="mse")
    NeuralPrefixGenome.random_init(k3, gc)

    base_eval = NeuralPrefixEvaluator(config=ec, data=(X, y))
    opt = optax.adam(learning_rate=0.1)

    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_eval, optimizer=opt, epochs=5)

    keys = jax.random.split(k3, 2)
    genes = jax.vmap(NeuralPrefixGenome.random_init, in_axes=(0, None))(keys, gc)
    from malthusjax.core.base import BasePopulation
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(2))

    initial_loss = base_eval.evaluate_population(pop, k4).fitness

    # Run Lamarckian update
    update = lsmf.evaluate_population(pop, k4)

    # Assert loss improved (or at least changed)
    assert not jnp.allclose(initial_loss, update.fitness)

    # Check that weights were modified
    assert not jnp.array_equal(pop.genes.weights, update.genes.weights)
    assert not jnp.array_equal(pop.genes.biases, update.genes.biases)

    # Check that topology remains the same
    assert jnp.array_equal(pop.genes.ops, update.genes.ops)
    assert jnp.array_equal(pop.genes.args, update.genes.args)

def test_lsmf_zero_epochs():
    """Verify LSMFEvaluator does not update weights when epochs=0."""
    key = jax.random.PRNGKey(222)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    X = jax.random.normal(k1, (20, 3))
    y = jax.random.normal(k2, (20,))

    gc = NeuralPrefixGenomeConfig(length=10, num_inputs=3, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=3, length=10, loss_function="mse")
    NeuralPrefixGenome.random_init(k3, gc)

    base_eval = NeuralPrefixEvaluator(config=ec, data=(X, y))
    opt = optax.adam(learning_rate=0.1)

    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_eval, optimizer=opt, epochs=0)

    keys = jax.random.split(k3, 2)
    genes = jax.vmap(NeuralPrefixGenome.random_init, in_axes=(0, None))(keys, gc)
    from malthusjax.core.base import BasePopulation
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(2))

    update = lsmf.evaluate_population(pop, k4)

    # Weights should be identical
    assert jnp.array_equal(pop.genes.weights, update.genes.weights)
    assert jnp.array_equal(pop.genes.biases, update.genes.biases)
