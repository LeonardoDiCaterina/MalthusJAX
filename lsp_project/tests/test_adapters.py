import jax
import jax.numpy as jnp
from lsp.adapters.cgp_adapter import build_cgp_engine
from lsp.adapters.mep_adapter import build_mep_engine
from lsp.adapters.dmep_adapter import build_dmep_engine
from lsp.adapters.lsmf_adapter import build_lsmf_engine
from malthusjax.core.genome.cartesian_genome import CartesianGenomeConfig
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
from lsp.genome.linear import PrefixGenomeConfig
from lsp.genome.neural_cartesian import NeuralCartesianGenome, NeuralCartesianGenomeConfig
from lsp.genome.neural_linear import NeuralPrefixGenome, NeuralPrefixGenomeConfig
from lsp.evaluator.cartesian import CartesianGPEvaluator, CartesianGPEvaluatorConfig
from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig

def get_dummy_data(key, num_samples=5, num_features=2):
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (num_samples, num_features))
    Y = jax.random.normal(k2, (num_samples, 1))
    return X, Y

def test_build_cgp_engine():
    key = jax.random.PRNGKey(42)
    X, Y = get_dummy_data(key)
    config = CartesianGenomeConfig(
        num_inputs=2, num_outputs=1, num_rows=2, num_cols=5, num_ops=4, max_arity=2, levels_back=3
    )
    eval_config = CartesianGPEvaluatorConfig(genome_config=config, batch_size=None)
    evaluator = CartesianGPEvaluator(config=eval_config, data=(X, Y))
    
    engine = build_cgp_engine(config, evaluator, pop_size=10, num_generations=2)
    assert engine is not None
    # Ensure it can init
    state = engine.init_state(key)
    assert state.population.fitness.shape == (10,)

def test_build_mep_engine():
    key = jax.random.PRNGKey(43)
    X, Y = get_dummy_data(key)
    config = PrefixGenomeConfig(
        num_inputs=2, length=10, num_ops=4, max_arity=2
    )
    eval_config = LinearGPEvaluatorConfig(num_inputs=2, length=10)
    evaluator = LinearGPEvaluator(config=eval_config, data=(X, Y))
    
    engine = build_mep_engine(config, evaluator, pop_size=10)
    assert engine is not None
    state = engine.init_state(key)
    assert state.population.fitness.shape == (10,)

def test_build_dmep_engine():
    key = jax.random.PRNGKey(44)
    X, Y = get_dummy_data(key)
    config = NeuralPrefixGenomeConfig(
        num_inputs=2, num_outputs=1, length=10, num_ops=4, max_arity=2, mep_output_strategy="dynamic"
    )
    eval_config = NeuralPrefixEvaluatorConfig(num_inputs=2, length=10)
    evaluator = NeuralPrefixEvaluator(config=eval_config, data=(X, Y))
    
    import optax
    optimizer = optax.adam(0.01)
    engine = build_dmep_engine(config, evaluator, optimizer, pop_size=10)
    assert engine is not None
    state = engine.init_state(key)
    assert state.population.fitness.shape == (10,)

def test_build_lsmf_engine():
    key = jax.random.PRNGKey(45)
    X, Y = get_dummy_data(key)
    config = NeuralCartesianGenomeConfig(
        num_inputs=2, num_outputs=1, num_rows=2, num_cols=5, num_ops=4, max_arity=2, levels_back=3
    )
    eval_config = NeuralCartesianEvaluatorConfig(genome_config=config)
    evaluator = NeuralCartesianEvaluator(config=eval_config, data=(X, Y))
    
    engine = build_lsmf_engine(config, evaluator, pop_size=10, num_generations=2)
    assert engine is not None
    state = engine.init_state(key)
    assert state.population.fitness.shape == (10,)

if __name__ == "__main__":
    test_build_cgp_engine()
    test_build_mep_engine()
    test_build_dmep_engine()
    test_build_lsmf_engine()
    print("Adapter tests passed!")
