import jax
import jax.numpy as jnp

from malthusjax.core.base import BasePopulation

# Core Genomes
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig

# Core Evaluators
from lsp.evaluator.cartesian import CartesianGPEvaluator, CartesianGPEvaluatorConfig
from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig

# Neural / Continuous Genomes
from lsp.genome.neural_cartesian import NeuralCartesianGenome, NeuralCartesianGenomeConfig
from lsp.genome.neural_linear import NeuralPrefixGenome, NeuralPrefixGenomeConfig

# Neural Evaluators
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.evaluator.differentiable_evaluator import (
    DifferentiableCartesianGPEvaluator,
    DifferentiableCartesianGPEvaluatorConfig,
)


def get_dummy_data(key, num_samples=10, num_features=3, num_targets=1, discrete_targets=False):
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (num_samples, num_features))
    if discrete_targets:
        Y = jax.random.randint(k2, (num_samples,), 0, num_targets)
    else:
        Y = jax.random.normal(k2, (num_samples, num_targets))
    return X, Y


def test_discrete_cgp():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    X, Y = get_dummy_data(k1, num_samples=5, num_features=2, num_targets=1)

    genome_config = CartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=4,
        max_arity=2,
        levels_back=3,
    )

    evaluator_config = CartesianGPEvaluatorConfig(
        genome_config=genome_config,
        batch_size=None,
    )

    evaluator = CartesianGPEvaluator(config=evaluator_config, data=(X, Y))

    # Initialize a population of 8 genomes
    genes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    # Evaluate the population
    evaluated_pop = evaluator.evaluate_population(pop, k3)

    assert evaluated_pop.fitness.shape == (8,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))


def test_discrete_mep():
    key = jax.random.PRNGKey(43)
    k1, k2, k3 = jax.random.split(key, 3)

    X, Y = get_dummy_data(k1, num_samples=5, num_features=2, num_targets=1)

    genome_config = LinearGenomeConfig(
        num_inputs=2,
        length=10,
        num_ops=4,
        max_arity=2,
    )

    evaluator_config = LinearGPEvaluatorConfig(
        num_inputs=2,
        length=10,
    )

    evaluator = LinearGPEvaluator(config=evaluator_config, data=(X, Y))

    genes = jax.vmap(LinearGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    evaluated_pop = evaluator.evaluate_population(pop, k3)

    assert evaluated_pop.fitness.shape == (8,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))


def test_continuous_dcgpann():
    key = jax.random.PRNGKey(44)
    k1, k2, k3 = jax.random.split(key, 3)

    X, Y = get_dummy_data(k1, num_samples=5, num_features=2, num_targets=1)

    genome_config = NeuralCartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=4,
        max_arity=2,
        levels_back=3,
    )

    evaluator_config = NeuralCartesianEvaluatorConfig(
        genome_config=genome_config,
    )

    evaluator = NeuralCartesianEvaluator(config=evaluator_config, data=(X, Y))

    genes = jax.vmap(NeuralCartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    evaluated_pop = evaluator.evaluate_population(pop, k3)

    assert evaluated_pop.fitness.shape == (8,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))


def test_continuous_dmep():
    key = jax.random.PRNGKey(45)
    k1, k2, k3 = jax.random.split(key, 3)

    X, Y = get_dummy_data(k1, num_samples=5, num_features=2, num_targets=1)

    genome_config = NeuralPrefixGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        length=10,
        num_ops=4,
        max_arity=2,
        mep_output_strategy="dynamic"
    )

    evaluator_config = NeuralPrefixEvaluatorConfig(
        num_inputs=2,
        length=10,
    )

    evaluator = NeuralPrefixEvaluator(config=evaluator_config, data=(X, Y))

    genes = jax.vmap(NeuralPrefixGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    evaluated_pop = evaluator.evaluate_population(pop, k3)

    assert evaluated_pop.fitness.shape == (8,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))


def test_pure_dcgp_evaluator():
    key = jax.random.PRNGKey(46)
    k1, k2, k3 = jax.random.split(key, 3)

    # Generate dummy data including Jacobian
    num_samples = 5
    num_features = 2
    X = jax.random.normal(k1, (num_samples, num_features))
    Y = jax.random.normal(k1, (num_samples, 1))
    dY_dX = jax.random.normal(k1, (num_samples, num_features))

    genome_config = CartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=11, # DEFAULT_DIFFERENTIABLE_OPS has 11 ops
        max_arity=2,
        levels_back=3,
    )

    evaluator_config = DifferentiableCartesianGPEvaluatorConfig(
        genome_config=genome_config,
        batch_size=None,
        grad_weight=1.0,
    )

    evaluator = DifferentiableCartesianGPEvaluator(config=evaluator_config, data=(X, Y, dY_dX))

    genes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    evaluated_pop = evaluator.evaluate_population(pop)

    assert evaluated_pop.fitness.shape == (8,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))
