import pytest
import jax
import jax.numpy as jnp
import optax
import chex

from malthusjax.core.genome.prefix.genome import ConstantGenomeConfig, ConstantAwarePrefixGenome
from malthusjax.core.fitness.prefix.linear_gp_prefix_evaluator import LinearGPPrefixEvaluatorConfig, LinearGPPrefixEvaluator
from malthusjax.operators.emitters.constant_optimizer import ConstantOptimizationEmitter

def test_constant_optimization_emitter():
    # 1. Dataset: y = 3.14 * x + 2.71
    # We will just evaluate MSE, we want the constants to be exactly 3.14 and 2.71
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 1), minval=-10, maxval=10)
    y = 3.14 * X[:, 0] + 2.71

    # 2. Configs
    genome_config = ConstantGenomeConfig(
        length=2, 
        num_inputs=1, 
        num_constants=2, 
        num_ops=6, # Standard ops, assume + and * are in here 
        max_arity=2
    )

    evaluator_config = LinearGPPrefixEvaluatorConfig(num_inputs=1, length=2)
    evaluator = LinearGPPrefixEvaluator(evaluator_config, (X, y))

    # 3. Create a mock population where the genome perfectly maps:
    # row 0: c0 * x
    # row 1: row 0 + c1
    # Ops: MULT=2, ADD=0 (assuming TENSORGP_FUNCTIONS)
    # TENSORGP_FUNCTIONS: 0=ADD, 1=SUB, 2=MUL...
    # So OP_MUL = 2, OP_ADD = 0
    # Args for row 0: [input(0), constant0(1)]
    # Args for row 1: [row0(3), constant1(2)]
    
    # We set initial constants arbitrarily
    k_pop = jax.random.PRNGKey(0)
    
    # We will hand-craft a genome and batch it
    ops = jnp.array([2, 0])
    args = jnp.array([
        [0, 1], # x_0 * c_0
        [3, 2]  # v_0 + c_1
    ])
    constants = jnp.array([1.0, 1.0]) # Starting far from 3.14 and 2.71
    
    base_genome = ConstantAwarePrefixGenome(ops=ops, args=args, constants=constants)
    
    # Create population of size 4
    batched_genome = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], 4, axis=0), base_genome)
    
    from malthusjax.core.genome.prefix.population import PrefixPopulation
    initial_pop = PrefixPopulation(genes=batched_genome, fitness=jnp.zeros(4), config=genome_config)
    initial_pop = evaluator.evaluate_population(initial_pop)
    
    print(f"Initial fitness: {initial_pop.fitness}")

    # 4. Run Emitter
    emitter = ConstantOptimizationEmitter(
        batch_size=4, 
        evaluator=evaluator,
        genome_config=genome_config,
        num_optimization_steps=200,
        learning_rate=0.1,
        emit_frequency=1
    )
    
    state = emitter.init(key, initial_pop)
    
    # The Emitter requires keys, we just pass dummy keys
    dummy_keys = jnp.zeros((4, 2))
    
    offspring_pop, new_state = emitter.ask(state, initial_pop, dummy_keys)
    
    print(f"Optimized constants: {offspring_pop.genes.constants}")
    
    # Verify constants hit targets
    c0 = offspring_pop.genes.constants[0, 0]
    c1 = offspring_pop.genes.constants[0, 1]
    
    print(f"c0 = {c0}, c1 = {c1}")
    
    assert jnp.allclose(c0, 3.14, atol=1e-2)
    assert jnp.allclose(c1, 2.71, atol=1e-2)

if __name__ == "__main__":
    test_constant_optimization_emitter()
