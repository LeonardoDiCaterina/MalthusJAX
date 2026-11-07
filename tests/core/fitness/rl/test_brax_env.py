"""Tests for Brax Environment Adapter (BraxEnv).

Verifies:
1. Compliance with EnvironmentComplianceSuite (dataclass, BaseRLEnvironment inheritance, obs_dim, action_dim).
2. Reset and step lifecycle contracts under pure JAX and jax.jit.
3. Continuous action postprocessing (tanh saturation).
4. End-to-end integration with RLEvaluator, MLPInterpreter, and ScalarOutput.
5. Population-level batch evaluation via jax.vmap.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.composable.base import (
    BaseEnvironment,
    IdentityTransform,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.environments import BraxEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.testing.compliance.environment_suite import EnvironmentComplianceSuite

brax = pytest.importorskip("brax")


class TestBraxEnvCompliance(EnvironmentComplianceSuite):
    """Verifies BraxEnv satisfies MalthusJAX EnvironmentComplianceSuite."""

    @pytest.fixture
    def component(self) -> BaseEnvironment:
        return BraxEnv.create("fast")


class TestBraxEnvLifecycle:
    """Tests reset, step, JIT-compilability and action saturation."""

    def test_brax_fast_reset_and_step_jit(self):
        env = BraxEnv.create("fast")
        assert env.obs_dim > 0
        assert env.action_dim > 0

        key = jax.random.PRNGKey(42)
        key_reset, key_step = jax.random.split(key)

        obs, state = jax.jit(env.reset)(key_reset)
        flat_obs = env.preprocess_obs(obs)
        assert flat_obs.shape == (env.obs_dim,)

        logits = jnp.full((env.action_dim,), 100.0)
        action = env.postprocess_action(logits)
        # tanh(100.0) == 1.0
        assert jnp.allclose(action, 1.0, atol=1e-5)

        next_obs, next_state, reward, done, info = jax.jit(env.step)(state, action, key_step)
        assert reward.shape == ()
        assert done.shape == ()

    def test_brax_postprocess_action_bounds(self):
        env = BraxEnv.create("fast")
        logits = jnp.array([-50.0, 50.0])
        action = env.postprocess_action(logits)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)


class TestBraxRLEvaluatorIntegration:
    """Tests BraxEnv inside the full composable evaluation stack."""

    def test_brax_fast_rl_evaluator_rollout(self):
        env = BraxEnv.create("fast")
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(8,),
            activation="tanh",
        )
        transform = IdentityTransform()
        output = ScalarOutput()

        evaluator = RLEvaluator(
            env=env,
            transform=transform,
            interpreter=interpreter,
            output=output,
            max_steps=20,
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

    def test_brax_population_vmap_evaluation(self):
        env = BraxEnv.create("fast")
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(8,),
            activation="tanh",
        )
        transform = IdentityTransform()
        output = ScalarOutput()

        evaluator = RLEvaluator(
            env=env,
            transform=transform,
            interpreter=interpreter,
            output=output,
            max_steps=10,
        )

        pop_size = 4
        n_params = interpreter.num_params
        config = RealGenomeConfig(shape=(n_params,))
        key = jax.random.PRNGKey(777)
        k_pop, k_eval = jax.random.split(key)
        weights = jax.random.normal(k_pop, (pop_size, n_params))
        pop = RealPopulation(
            genes=RealGenome(values=weights),
            fitness=jnp.full((pop_size,), jnp.nan),
            config=config,
        )

        def eval_single(gene_vals: chex.Array, subkey: chex.PRNGKey):
            gene = RealGenome(values=gene_vals)
            compiled = transform.transform(gene)
            fit, _ = evaluator.evaluate(raw_genome=gene, compiled_genome=compiled, rng=subkey)
            return fit

        keys = jax.random.split(k_eval, pop_size)
        vmapped_eval = jax.jit(jax.vmap(eval_single))
        fitnesses = vmapped_eval(pop.genes.values, keys)

        assert fitnesses.shape == (pop_size,)
        assert jnp.all(jnp.isfinite(fitnesses))
