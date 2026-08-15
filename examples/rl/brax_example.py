"""Example of using the BraxEvaluator in MalthusJAX."""

import jax
import jax.numpy as jnp

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.brax_evaluator import BraxEvaluator, BraxEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


def run_brax_eval():
    print("Initializing Brax Ant evaluator...")
    config = BraxEvaluatorConfig(env_name="ant", max_steps=100)
    evaluator = BraxEvaluator.create(config)

    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), jnp.zeros((1, evaluator.env.observation_size)))
    )
    genome_size = sum(p.size for p in flat_params)
    print(f"Required genome size: {genome_size}")

    pop_size = 10
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size)) * 0.1
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    print("Evaluating population (this may take a moment to JIT compile physics)...")
    jit_eval_pop = jax.jit(evaluator.evaluate_population)

    evaluated_pop = jit_eval_pop(population)

    print("Evaluation complete!")
    print(f"Fitness scores: {evaluated_pop.fitness}")
    print(f"Best fitness: {jnp.max(evaluated_pop.fitness)}")


if __name__ == "__main__":
    run_brax_eval()
