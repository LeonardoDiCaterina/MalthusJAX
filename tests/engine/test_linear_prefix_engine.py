"""End-to-end test of the Linear GP Prefix topology in the GeneticFastEngine."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.prefix.linear_gp_prefix_evaluator import (
    LinearGPPrefixEvaluator,
)
from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngineParams, GeneticEngine, OperatorState
from malthusjax.operators.crossover.linear import LinearUniformCrossover
from malthusjax.operators.mutation.linear import LinearMutation
from malthusjax.operators.selection.prefix.tournament import PrefixTournamentSelection


def test_linear_prefix_engine_integration():
    """Run an end-to-end evolutionary loop with the prefix architecture.
    
    This verifies that all the pieces (Genome, Evaluator, Selection, Mutation, Crossover)
    plug into the standard GeneticFastEngine perfectly and can optimize a toy problem.
    """
    # 1. Toy Problem: Learn y = x0 * x1 + x2
    # We will provide 3 inputs.
    # Operations: 0: ADD, 1: SUB, 2: MUL, 3: DIV
    
    def toy_fitness_fn(args: jnp.ndarray, ops: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        """A simple non-JIT interpreter for testing."""
        # args: (L, 2), ops: (L,)
        L = ops.shape[0]
        N = inputs.shape[0]
        buffer = jnp.zeros(N + L)
        buffer = buffer.at[:N].set(inputs)
        
        for i in range(L):
            a = buffer[args[i, 0]]
            b = buffer[args[i, 1]]
            op = ops[i]
            
            # Simple ops
            res = jax.lax.switch(
                op,
                [
                    lambda a, b: a + b,
                    lambda a, b: a - b,
                    lambda a, b: a * b,
                    lambda a, b: jnp.where(b != 0, a / b, 1.0)
                ],
                a, b
            )
            buffer = buffer.at[N + i].set(res)
            
        # Return all prefix evaluations: buffer[N:]
        return buffer[N:]

    # Our batched fitness function expects (pop_size, L) prefix evaluations.
    # We will evaluate against a few fixed input/target pairs.
    # Targets for y = x0 * x1 + x2
    test_cases = jnp.array([
        [1.0, 2.0, 3.0], # 1*2+3 = 5
        [0.0, 5.0, 1.0], # 0*5+1 = 1
        [-2.0, 3.0, 4.0], # -2*3+4 = -2
    ])
    targets = jnp.array([5.0, 1.0, -2.0])
    
    def evaluate_genome_prefixes(genome, config):
        """Returns MSE for each prefix row. Shape (L,)"""
        def _eval_one_case(inputs):
            return toy_fitness_fn(genome.args, genome.ops, inputs)
            
        preds = jax.vmap(_eval_one_case)(test_cases) # (num_cases, L)
        # MSE against targets (broadcasting targets to (num_cases, 1))
        mse = jnp.mean((preds - targets[:, None]) ** 2, axis=0) # (L,)
        # We want to maximize fitness, so return -MSE
        return -mse
        
    class ToyPrefixEvaluator(LinearGPPrefixEvaluator):
        def _evaluate_batch(self, batched_genes, config, keys, **kwargs):
            # vmap over population
            fitness = jax.vmap(evaluate_genome_prefixes, in_axes=(0, None))(batched_genes, config)
            return fitness

    # 2. Configurations
    L = 5
    pop_size = 32
    num_generations = 10
    
    genome_config = PrefixGenomeConfig(
        length=L, num_inputs=3, num_ops=4, max_arity=2
    )
    
    # 3. Operators
    crossover = LinearUniformCrossover(num_offspring=1, crossover_rate=0.5)
    
    mutation = LinearMutation(mutation_rate=0.2, p_internal=0.5, decay_name="uniform")
    
    selection = PrefixTournamentSelection(num_selections=pop_size, tournament_size=3, n_elites=1)
    from malthusjax.core.fitness.prefix.linear_gp_prefix_evaluator import LinearGPPrefixEvaluatorConfig
    evaluator_config = LinearGPPrefixEvaluatorConfig(num_inputs=3, length=L)
    dummy_data = (jnp.zeros(1), jnp.zeros(1))
    evaluator = ToyPrefixEvaluator(config=evaluator_config, data=dummy_data)
    
    # 4. Engine
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=num_generations,
        elitism=1,
    )
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params,
    )
    
    # 5. Run the engine manually for one step to verify JIT and shapes
    key = jax.random.PRNGKey(42)
    state = engine.init_state(key)
    
    # We will just compile and run one step
    @jax.jit
    def jit_step(s):
        return engine.step(s)
        
    state, metrics = jit_step(state)
    
    # 6. Verify
    # Extract the population from the final state
    pop = state.population
    
    # Ensure population shapes are correct
    assert pop.genes.ops.shape == (pop_size, L)
    assert pop.genes.args.shape == (pop_size, L, 2)
    assert pop.fitness.shape == (pop_size,)
    assert pop.prefix_fitness.shape == (pop_size, L)
    
    # The best fitness should be better (closer to 0) than -infinity or very bad random
    # Actually, a toy problem of this size might solve it perfectly in 10 generations.
    best_fitness = jnp.max(pop.fitness)
    assert best_fitness > -100.0 # Better than a terrible random guess
    
    # Structural validity constraint
    for i in range(L):
        max_valid_idx = 3 + i - 1
        assert jnp.all(pop.genes.args[:, i] <= max_valid_idx)
        assert jnp.all(pop.genes.args[:, i] >= 0)
        
if __name__ == "__main__":
    test_linear_prefix_engine_integration()
