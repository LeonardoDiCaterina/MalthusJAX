"""Example of using the JumanjiEvaluator in MalthusJAX."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.rl.jumanji_evaluator import JumanjiEvaluator, JumanjiEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.base import BasePopulation

def run_jumanji_eval():
    print("Initializing Jumanji Snake evaluator...")
    config = JumanjiEvaluatorConfig(env_name="Snake-v1", max_steps=50)
    evaluator = JumanjiEvaluator.create(config)
    
    _, dummy_timestep = evaluator.env.reset(jax.random.PRNGKey(0))
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), dummy_timestep.observation)
    )
    genome_size = sum(p.size for p in flat_params)
    print(f"Required genome size: {genome_size}")
    
    pop_size = 10
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)
    
    print("Evaluating population...")
    jit_eval_pop = jax.jit(evaluator.evaluate_population)
    
    evaluated_pop = jit_eval_pop(population)
    
    print("Evaluation complete!")
    print(f"Fitness scores: {evaluated_pop.fitness}")
    print(f"Best fitness: {jnp.max(evaluated_pop.fitness)}")

if __name__ == "__main__":
    run_jumanji_eval()
