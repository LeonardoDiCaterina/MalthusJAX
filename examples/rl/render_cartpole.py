"""Script to train and visualize a CartPole agent using MalthusJAX."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.environments import GymnaxEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealPopulation, RealGenomeConfig


def train_and_render():
    print("Initializing Gymnax CartPole evaluator...")
    env_name = "CartPole-v1"
    env = GymnaxEnv.create(env_name=env_name)
    interpreter = MLPInterpreter(input_dim=env.obs_dim, output_dim=env.action_dim, hidden=(32,))
    output = ScalarOutput(maximize=True)
    evaluator = RLEvaluator(env=env, interpreter=interpreter, output=output, max_steps=500, num_eval_envs=10)

    genome_size = interpreter.num_params
    print(f"Required genome size: {genome_size}")

    print("Running random search to find a robust policy...")
    pop_size = 200
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    config = RealGenomeConfig(shape=(genome_size,))
    population = RealPopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=config)

    evaluated_pop = evaluator.evaluate_population(population, rng=jax.random.PRNGKey(101))

    best_idx = jnp.argmax(evaluated_pop.fitness)
    best_fitness = evaluated_pop.fitness[best_idx]
    print(f"Best robust fitness: {best_fitness}")


if __name__ == "__main__":
    train_and_render()
