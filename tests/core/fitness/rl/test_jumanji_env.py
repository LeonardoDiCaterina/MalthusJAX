"""Tests for Jumanji Environment Adapter (JumanjiEnv).

Verifies:
1. Compliance with EnvironmentComplianceSuite (dataclass, BaseRLEnvironment inheritance, obs_dim, action_dim).
2. Reset and step lifecycle contracts under pure JAX and jax.jit.
3. Action masking invariant: postprocess_action masks illegal actions even with large logits (+1e6).
4. Adversarial violation test: executing an invalid action sets done=True without crashing or NaN.
5. End-to-end integration with RLEvaluator, MLPInterpreter, and ScalarOutput.
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
from malthusjax.core.fitness.composable.environments import JumanjiEnv
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.testing.compliance.environment_suite import EnvironmentComplianceSuite

jumanji = pytest.importorskip("jumanji")


class TestJumanjiEnvCompliance(EnvironmentComplianceSuite):
    """Verifies JumanjiEnv satisfies MalthusJAX EnvironmentComplianceSuite."""

    @pytest.fixture
    def component(self) -> BaseEnvironment:
        return JumanjiEnv.create("Knapsack-v1")


class TestJumanjiEnvLifecycle:
    """Tests reset, step, JIT-compilability and masking invariants."""

    def test_jumanji_reset_and_step_jit(self):
        env = JumanjiEnv.create("Knapsack-v1")
        assert env.obs_dim > 0
        assert env.action_dim > 0

        key = jax.random.PRNGKey(42)
        key_reset, key_step = jax.random.split(key)

        timestep, state = jax.jit(env.reset)(key_reset)
        flat_obs = env.preprocess_obs(timestep)
        assert flat_obs.shape == (env.obs_dim,)

        action = jnp.int32(0)
        next_timestep, next_state, reward, done, info = jax.jit(env.step)(state, action, key_step)
        assert reward.shape == ()
        assert done.shape == ()
        assert isinstance(done, (bool, jnp.bool_, jax.Array))

    def test_jumanji_action_mask_blocks_illegal_high_logits(self):
        """Metamorphic test: illegal actions with massive logits (+1e6) must never be chosen."""
        env = JumanjiEnv.create("Knapsack-v1")
        action_dim = env.action_dim

        # Construct logits with huge illegal spikes at indices 0 and 1
        logits = jnp.full((action_dim,), -10.0)
        logits = logits.at[0].set(1e6)
        logits = logits.at[1].set(1e6)
        logits = logits.at[2].set(-5.0)  # The highest legal action

        # Mask allowing only index 2
        mask = jnp.zeros(action_dim, dtype=bool).at[2].set(True)

        class MockObs:
            action_mask = mask

        class MockTimeStep:
            observation = MockObs()

        chosen_action = env.postprocess_action(logits, MockTimeStep())
        assert chosen_action == 2, f"Expected action index 2, got {chosen_action}"

    def test_jumanji_adversarial_illegal_action_step(self):
        """Adversarial test: taking an illegal action terminates gracefully (done=True)."""
        env = JumanjiEnv.create("Knapsack-v1")
        key = jax.random.PRNGKey(999)
        timestep, state = env.reset(key)

        # Legally pack item 0
        action = jnp.int32(0)
        ts1, s1, r1, done1, _ = env.step(state, action, key)
        assert not done1

        # Item 0 is already packed and now illegal. Pack item 0 again!
        ts2, s2, r2, done2, _ = jax.jit(env.step)(s1, action, key)
        assert done2, "Adversarial illegal action did not trigger done=True"
        assert jnp.isfinite(r2), f"Reward must be finite, got {r2}"


class TestJumanjiRLEvaluatorIntegration:
    """Tests JumanjiEnv inside the full composable evaluation stack."""

    def test_knapsack_rl_evaluator_rollout(self):
        env = JumanjiEnv.create("Knapsack-v1")
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(16,),
            activation="relu",
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
