import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
import numpy as np

# --- IMPORTS ---
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation
from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig
from malthusjax.operators.crossover.real import BlendCrossover, SimulatedBinaryCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

# ==============================================================================
# 1. LINEAR GP TESTS
# ==============================================================================

@pytest.fixture
def linear_config():
    """Standard config for Linear GP tests."""
    return LinearGenomeConfig(
        length=10,
        num_inputs=2,
        num_ops=5,
        max_arity=3
    )

@pytest.fixture
def linear_genome(linear_config):
    """Creates a deterministic LinearGenome."""
    key = jar.PRNGKey(42)
    return LinearGenome.random_init(key, linear_config)

def test_linear_genome_methods(linear_config, linear_genome):
    """Test helper methods in LinearGenome."""
    # 1. Test Autocorrect
    broken_genome = linear_genome.replace(
        args=jnp.full_like(linear_genome.args, 999) 
    )
    corrected = broken_genome.autocorrect(linear_config)
    assert jnp.all(corrected.args < (linear_config.num_inputs + linear_config.length))

    # 2. Test Distance
    dist = linear_genome.distance(broken_genome)
    assert dist > 0

    # 3. Test Render
    op_names = ["ADD", "SUB", "MUL", "DIV", "SIN"]
    render_str = linear_genome.render(linear_config, op_names=op_names)
    assert "v_" in render_str

    # 4. Test Population Init
    pop = LinearPopulation.init_random(jar.PRNGKey(0), linear_config, size=5)
    assert pop.genes.ops.shape == (5, linear_config.length)

def test_linear_evaluator_execution(linear_config, linear_genome):
    """Test LinearGPEvaluator execution logic."""
    X = jnp.ones((5, linear_config.num_inputs))
    y = jnp.ones(5)
    
    # FIX: Added maximize=True
    eval_config = LinearGPEvaluatorConfig(
        X=X, y=y, 
        num_inputs=linear_config.num_inputs, 
        length=linear_config.length,
        maximize=True
    )
    evaluator = LinearGPEvaluator(config=eval_config, data=(X, y))

    # 1. Test Single Prediction
    single_out = evaluator.predict_one(linear_genome, X[0])
    assert single_out.shape == (linear_config.length,)

    # 2. Test Full Evaluation
    fitness = evaluator.evaluate(linear_genome)
    assert isinstance(fitness, (float, jnp.ndarray))
    assert fitness.ndim == 0 

    # 3. Test program output extraction
    prog_out = evaluator.get_program_prediction(linear_genome, X, instruction_idx=-1)
    assert prog_out.shape == (5,)

# ==============================================================================
# 2. REAL CROSSOVER TESTS
# ==============================================================================

def test_real_crossover_operators():
    """Test Blend and SBX Crossover."""
    key = jar.PRNGKey(101)
    # FIX: Used bounds tuple instead of min_value/max_value
    config = RealGenomeConfig(length=5, bounds=(0.0, 1.0))
    
    p1 = RealGenome(values=jnp.zeros(5))
    p2 = RealGenome(values=jnp.ones(5))

    # 1. Test Blend Crossover
    blx = BlendCrossover(crossover_rate=1.0, alpha=0.5)
    child_blx = blx._cross_one(key, p1, p2, config)
    assert not jnp.all(child_blx.values == p1.values)
    
    # 2. Test SBX Crossover
    sbx = SimulatedBinaryCrossover(crossover_rate=1.0, eta=20.0)
    child_sbx = sbx._cross_one(key, p1, p2, config)
    assert child_sbx.values.shape == (5,)
    assert not jnp.all(child_sbx.values == p1.values)

    # 3. Test No Crossover
    blx_none = BlendCrossover(crossover_rate=0.0)
    child_none = blx_none._cross_one(key, p1, p2, config)
    assert jnp.array_equal(child_none.values, p1.values)