"""Script to train and visualize a Jumanji agent using MalthusJAX."""

import os

import jax
import jax.numpy as jnp

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.jumanji_evaluator import JumanjiEvaluator, JumanjiEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


def train_and_render():
    print("Initializing Jumanji Snake evaluator...")
    env_name = "Snake-v1"
    config = JumanjiEvaluatorConfig(env_name=env_name, max_steps=100, num_eval_envs=10)
    evaluator = JumanjiEvaluator.create(config)

    _, dummy_timestep = evaluator.env.reset(jax.random.PRNGKey(0))
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), dummy_timestep.observation)
    )
    genome_size = sum(p.size for p in flat_params)
    print(f"Required genome size: {genome_size}")

    print("Running random search to find a decent policy...")
    pop_size = 2000
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)

    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    jit_eval_pop = jax.jit(evaluator.evaluate_population)
    evaluated_pop = jit_eval_pop(population)

    best_idx = jnp.argmax(evaluated_pop.fitness)
    best_fitness = evaluated_pop.fitness[best_idx]
    best_genome_vals = evaluated_pop.genes.values[best_idx]
    print(f"Best robust fitness: {best_fitness}")

    print("Unrolling best policy to collect states and render...")
    policy_params = evaluator.unflatten_fn(best_genome_vals)
    env = evaluator.env

    rng, reset_rng = jax.random.split(rng)
    state, timestep = env.reset(reset_rng)

    # We must save states for Jumanji's native animation
    states = [state]

    for _ in range(config.max_steps):
        # Add batch dim for policy. Jumanji obs are pytrees (e.g. action_mask, step_count)
        obs_batch = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), timestep.observation
        )

        action_logits = evaluator.policy.apply(policy_params, obs_batch)
        action = jnp.argmax(action_logits, axis=-1)[0]

        state, timestep = env.step(state, action)
        states.append(state)

        if timestep.last():
            break

    print(f"Episode collected. Length: {len(states)}")

    print("Saving animation to examples/rl/videos/snake_robust_policy.gif...")
    os.makedirs("examples/rl/videos", exist_ok=True)
    env.animate(states, interval=150, save_path="examples/rl/videos/snake_robust_policy.gif")
    print("Done! You can view the animation at examples/rl/videos/snake_robust_policy.gif")


if __name__ == "__main__":
    train_and_render()
