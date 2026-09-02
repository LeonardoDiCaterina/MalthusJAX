"""Tests for QD Evaluator and Emitter Integration in LSP."""

import jax
import jax.numpy as jnp
import pytest
from lsp.evaluator.differentiable_evaluator import DifferentiableCartesianGPEvaluatorConfig
from lsp.evaluator.qd_evaluator import DifferentiableCartesianQDEvaluator
from lsp.operators.qd_emitter import LSPQDEmitter
from lsp.operators.cartesian import CartesianMicroMutation
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.core.base import BasePopulation
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams

# Ensure QDAX is installed for these tests
try:
    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
    HAS_QDAX = True
except ImportError:
    HAS_QDAX = False


@pytest.mark.skipif(not HAS_QDAX, reason="QDAX is required for MAP-Elites engine tests")
def test_pure_dcgp_qd_evaluator():
    """Test that DifferentiableCartesianQDEvaluator correctly computes fitness and BDs."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

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
        num_ops=11,
        max_arity=2,
        levels_back=3,
    )

    evaluator_config = DifferentiableCartesianGPEvaluatorConfig(
        genome_config=genome_config,
        batch_size=None,
        grad_weight=1.0,
    )

    evaluator = DifferentiableCartesianQDEvaluator(config=evaluator_config, data=(X, Y, dY_dX))

    genes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    qd_pop = evaluator.evaluate_population(pop)

    assert qd_pop.fitness.shape == (8,)
    assert "descriptors" in qd_pop.info
    assert qd_pop.info["descriptors"].shape == (8, 2)  # Active Nodes, Variance


@pytest.mark.skipif(not HAS_QDAX, reason="QDAX is required for MAP-Elites engine tests")
def test_qd_engine_integration():
    """Test the full MapElitesEngine with LSPQDEmitter and DifferentiableCartesianQDEvaluator."""
    key = jax.random.PRNGKey(101)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    num_samples = 5
    num_features = 2
    X = jax.random.normal(k1, (num_samples, num_features))
    Y = jax.random.normal(k1, (num_samples, 1))
    dY_dX = jax.random.normal(k1, (num_samples, num_features))

    genome_config = CartesianGenomeConfig(
        num_inputs=2, num_outputs=1, num_rows=2, num_cols=5, num_ops=11, max_arity=2, levels_back=3
    )
    
    genes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 20), genome_config
    )
    initial_pop = BasePopulation(genes=genes, fitness=jnp.zeros(20), config=genome_config)

    evaluator = DifferentiableCartesianQDEvaluator(
        config=DifferentiableCartesianGPEvaluatorConfig(
            genome_config=genome_config, grad_weight=1.0
        ),
        data=(X, Y, dY_dX)
    )

    mutation_op = CartesianMicroMutation(mutation_rate=0.2)
    # Using 100% mutation since pure CGP doesn't have a crossover yet
    emitter = LSPQDEmitter(
        mutation_op=mutation_op,
        variation_op=None,
        variation_percentage=0.0,
        _batch_size=20,
        genome_config=genome_config
    )

    engine_params = MapElitesEngineParams(maximize=True) # QD Engine Native

    engine = MapElitesEngine(emitter=emitter, evaluator=evaluator, engine_params=engine_params)

    # Fake centroids for a 10x10 grid (100 bins)
    # BD1: Active Nodes (say 1 to 10), BD2: Variance (say 0.0 to 1.0)
    bd1 = jnp.linspace(1, 10, 10)
    bd2 = jnp.linspace(0.0, 1.0, 10)
    c1, c2 = jnp.meshgrid(bd1, bd2)
    centroids = jnp.stack([c1.flatten(), c2.flatten()], axis=-1)

    state = engine.init_state(k3, initial_pop, centroids)
    
    assert state.repertoire is not None
    
    # Run one step
    state, metrics = engine.step(state)
    
    assert metrics.generation == 1
    assert metrics.coverage >= 0.0


def test_linear_qd_evaluator():
    from lsp.evaluator.qd_evaluator import DifferentiableLinearQDEvaluator
    from lsp.evaluator.differentiable_evaluator import DifferentiableLinearGPEvaluatorConfig
    from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    num_samples = 5
    num_features = 2
    X = jax.random.normal(k1, (num_samples, num_features))
    Y = jax.random.normal(k1, (num_samples, 1))
    dY_dX = jax.random.normal(k1, (num_samples, num_features))

    genome_config = LinearGenomeConfig(
        num_inputs=2,
        length=10,
        num_ops=11,
        max_arity=2,
    )

    evaluator_config = DifferentiableLinearGPEvaluatorConfig(
        num_inputs=2,
        length=10,
        grad_weight=1.0,
    )

    evaluator = DifferentiableLinearQDEvaluator(config=evaluator_config, data=(X, Y, dY_dX))

    genes = jax.vmap(LinearGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    qd_pop = evaluator.evaluate_population(pop)

    assert qd_pop.fitness.shape == (8,)
    assert "descriptors" in qd_pop.info
    assert qd_pop.info["descriptors"].shape == (8, 2)  # Active Depth, Variance


def test_lsmf_qd_evaluator():
    from lsp.evaluator.qd_evaluator import LSMFQDEvaluator
    from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig
    from lsp.genome.neural_cartesian import NeuralCartesianGenome, NeuralCartesianGenomeConfig
    import optax
    
    key = jax.random.PRNGKey(43)
    k1, k2 = jax.random.split(key)

    num_samples = 5
    num_features = 2
    X = jax.random.normal(k1, (num_samples, num_features))
    Y = jax.random.normal(k1, (num_samples, 1))

    genome_config = NeuralCartesianGenomeConfig(
        num_inputs=2,
        num_outputs=1,
        num_rows=2,
        num_cols=5,
        num_ops=11,
        max_arity=2,
        levels_back=3,
    )

    base_evaluator_config = NeuralCartesianEvaluatorConfig(
        genome_config=genome_config,
    )
    base_evaluator = NeuralCartesianEvaluator(config=base_evaluator_config, data=(X, Y))

    evaluator = LSMFQDEvaluator(
        config=None, 
        data=(X, Y), 
        base_evaluator=base_evaluator, 
        optimizer=optax.adam(0.01), 
        epochs=5
    )

    genes = jax.vmap(NeuralCartesianGenome.random_init, in_axes=(0, None))(
        jax.random.split(k2, 8), genome_config
    )
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(8))

    qd_pop = evaluator.evaluate_population(pop)

    assert qd_pop.fitness.shape == (8,)
    assert "descriptors" in qd_pop.info
    assert qd_pop.info["descriptors"].shape == (8, 2)  # Active Nodes, Final Training Loss
