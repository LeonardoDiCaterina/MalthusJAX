"""Script to train and visualize a CartPole agent using MalthusJAX."""

import os

import jax
import jax.numpy as jnp

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.gymnax_evaluator import GymnaxEvaluator, GymnaxEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


def train_and_render():
    print("Initializing Gymnax CartPole evaluator...")
    env_name = "CartPole-v1"
    # Use 10 eval envs to force the random search to find a generalized policy
    config = GymnaxEvaluatorConfig(env_name=env_name, max_steps=500, num_eval_envs=10)
    evaluator = GymnaxEvaluator.create(config)

    # Infer genome size
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1,) + evaluator.env.observation_space(evaluator.env_params).shape),
        )
    )
    genome_size = sum(p.size for p in flat_params)
    print(f"Required genome size: {genome_size}")

    print("Running random search to find a robust policy...")
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
    print(f"Best robust fitness (avg over 10 rollouts): {best_fitness}")

    print("Unrolling best policy to collect states and render...")
    policy_params = evaluator.unflatten_fn(best_genome_vals)

    try:
        import gymnasium as gym
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        import gym
        from gym.wrappers import RecordVideo

    # Use rgb_array to allow the wrapper to capture frames
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Wrap the environment to save the video

    os.makedirs("examples/rl/videos", exist_ok=True)
    render_env = RecordVideo(
        render_env,
        video_folder="examples/rl/videos",
        name_prefix="cartpole_robust_policy",
        episode_trigger=lambda x: True,
    )

    for episode in range(5):
        # Handle both new API (obs, info) and old API (obs)
        reset_out = render_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        steps = 0
        while steps < config.max_steps:
            # We need to reshape obs exactly as evaluate does
            obs_batch = jnp.expand_dims(jnp.array(obs), axis=0)
            action = evaluator.policy.apply(policy_params, obs_batch)

            # Discrete space, sample or argmax
            action_idx = int(jnp.argmax(action, axis=-1)[0])

            step_out = render_env.step(action_idx)
            # Handle both new API (obs, reward, term, trunc, info) and old (obs, reward, done, info)
            obs = step_out[0]
            done = step_out[2] if len(step_out) == 4 else (step_out[2] or step_out[3])

            steps += 1
            if done:
                break

        print(f"Episode {episode} rendered. Length: {steps}")

    render_env.close()


if __name__ == "__main__":
    train_and_render()
