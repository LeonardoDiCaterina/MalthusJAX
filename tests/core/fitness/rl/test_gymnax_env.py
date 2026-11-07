"""Tests for Gymnax Environment Adapter (GymnaxEnv).

Verifies:
1. Compliance with EnvironmentComplianceSuite (dataclass, BaseRLEnvironment inheritance, obs_dim, action_dim).
2. Reset and step lifecycle contracts under pure JAX and jax.jit.
3. Discrete (CartPole-v1) and continuous (Pendulum-v1) action postprocessing.
4. End-to-end integration with RLEvaluator, MLPInterpreter, and ScalarOutput.
5. Population-level batch evaluation via jax.vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.composable.base import (
    BaseEnvironment,
    IdentityTransform,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.environments import GymnaxEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.testing.compliance.environment_suite import EnvironmentComplianceSuite

gymnax = pytest.importorskip("gymnax")


class TestGymnaxEnvCompliance(EnvironmentComplianceSuite):
    """Verifies GymnaxEnv satisfies MalthusJAX EnvironmentComplianceSuite."""

    @pytest.fixture
    def component(self) -> BaseEnvironment:
        return GymnaxEnv.create("CartPole-v1")


class TestGymnaxEnvLifecycle:
    """Tests reset, step, JIT-compilability and continuous/discrete spaces."""

    def test_cartpole_reset_and_step_jit(self):
        env = GymnaxEnv.create("CartPole-v1")
        assert env.obs_dim == 4
        assert env.action_dim == 2

        key = jax.random.PRNGKey(42)
        key_reset, key_step = jax.random.split(key)

        obs, state = jax.jit(env.reset)(key_reset)
        flat_obs = env.preprocess_obs(obs)
        assert flat_obs.shape == (4,)

        logits = jnp.array([-10.0, 10.0])
        action = env.postprocess_action(logits)
        assert action == 1

        next_obs, next_state, reward, done, info = jax.jit(env.step)(state, action, key_step)
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(done, (bool, jnp.bool_, jax.Array))

    def test_pendulum_continuous_action(self):
        env = GymnaxEnv.create("Pendulum-v1")
        assert env.obs_dim == 3
        assert env.action_dim == 1

        key = jax.random.PRNGKey(123)
        obs, state = env.reset(key)
        flat_obs = env.preprocess_obs(obs)
        assert flat_obs.shape == (3,)

        logits = jnp.array([1.5])
        action = env.postprocess_action(logits)
        assert jnp.allclose(action, logits)

        next_obs, next_state, reward, done, info = env.step(state, action, key)
        assert reward.shape == ()
        assert not done


class TestGymnaxRLEvaluatorIntegration:
    """Tests GymnaxEnv inside the full composable evaluation stack."""

    def test_cartpole_rl_evaluator_rollout(self):
        env = GymnaxEnv.create("CartPole-v1")
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(16,),
            activation="tanh",
        )
        transform = IdentityTransform()
        output = ScalarOutput(maximize=True)

        evaluator = RLEvaluator(
            env=env,
            transform=transform,
            interpreter=interpreter,
            output=output,
            max_steps=50,
        )

        n_params = interpreter.num_params
        genome = RealGenome(values=jnp.zeros(n_params))
        rng = jax.random.PRNGKey(0)

        fitness, metrics = evaluator.evaluate(
            raw_genome=genome,
            compiled_genome=transform.transform(genome),
            rng=rng,
        )

        assert fitness.shape == ()
        assert jnp.isfinite(fitness)
        assert fitness < 0.0  # CartPole achieves positive reward, so maximize -> negative fitness

    def test_gymnax_population_vmap_evaluation(self):
        env = GymnaxEnv.create("CartPole-v1")
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(8,),
            activation="tanh",
        )
        transform = IdentityTransform()
        output = ScalarOutput(maximize=True)

        evaluator = RLEvaluator(
            env=env,
            transform=transform,
            interpreter=interpreter,
            output=output,
            max_steps=30,
        )

        pop_size = 8
        n_params = interpreter.num_params
        config = RealGenomeConfig(shape=(n_params,))
        key = jax.random.PRNGKey(99)
        k_pop, k_eval = jax.random.split(key)
        weights = jax.random.normal(k_pop, (pop_size, n_params))
        pop = RealPopulation(
            genes=RealGenome(values=weights),
            fitness=jnp.full((pop_size,), jnp.nan),
            config=config,
        )

        def eval_single(gene_vals: jax.Array, subkey: jax.Array):
            gene = RealGenome(values=gene_vals)
            compiled = transform.transform(gene)
            fit, _ = evaluator.evaluate(raw_genome=gene, compiled_genome=compiled, rng=subkey)
            return fit

        keys = jax.random.split(k_eval, pop_size)
        vmapped_eval = jax.jit(jax.vmap(eval_single))
        fitnesses = vmapped_eval(pop.genes.values, keys)

        assert fitnesses.shape == (pop_size,)
        assert jnp.all(jnp.isfinite(fitnesses))
