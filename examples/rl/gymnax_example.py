"""Example of using the composable Gymnax environment adapter in MalthusJAX."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.environments import GymnaxEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealPopulation, RealGenomeConfig


def run_gymnax_eval():
    print("Initializing Gymnax CartPole composable evaluator...")
    env = GymnaxEnv.create(env_name="CartPole-v1")
    interpreter = MLPInterpreter(input_dim=env.obs_dim, output_dim=env.action_dim, hidden=(32,))
    output = ScalarOutput(maximize=True)
    evaluator = RLEvaluator(env=env, interpreter=interpreter, output=output, max_steps=200, num_eval_envs=1)

    genome_size = interpreter.num_params
    print(f"Required genome size: {genome_size}")

    # Create a random population of 10 genomes
    pop_size = 10
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    config = RealGenomeConfig(shape=(genome_size,))
    population = RealPopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=config)

    print("Evaluating population...")
    # Evaluate population
    evaluated_pop = evaluator.evaluate_population(population, rng=jax.random.PRNGKey(123))

    print("Evaluation complete!")
    print(f"Fitness scores: {evaluated_pop.fitness}")
    print(f"Best fitness: {jnp.max(evaluated_pop.fitness)}")


if __name__ == "__main__":
    run_gymnax_eval()
