"""Example of using the GymnaxEvaluator in MalthusJAX."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.rl.gymnax_evaluator import GymnaxEvaluator, GymnaxEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.base import BasePopulation

def run_gymnax_eval():
    print("Initializing Gymnax CartPole evaluator...")
    config = GymnaxEvaluatorConfig(env_name="CartPole-v1", max_steps=200, num_eval_envs=1)
    evaluator = GymnaxEvaluator.create(config)
    
    # Infer genome size from dummy params
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), jnp.zeros((1,) + evaluator.env.observation_space(evaluator.env_params).shape))
    )
    genome_size = sum(p.size for p in flat_params)
    print(f"Required genome size: {genome_size}")
    
    # Create a random population of 10 genomes
    pop_size = 10
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)
    
    print("Evaluating population...")
    # JIT compile the population evaluation
    jit_eval_pop = jax.jit(evaluator.evaluate_population)
    
    # Run evaluation
    evaluated_pop = jit_eval_pop(population)
    
    print("Evaluation complete!")
    print(f"Fitness scores: {evaluated_pop.fitness}")
    print(f"Best fitness: {jnp.max(evaluated_pop.fitness)}")

if __name__ == "__main__":
    run_gymnax_eval()
