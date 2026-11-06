"""Tests for DeepMind MuJoCo MJX Environment Adapter (MuJoCoMJXEnv).

Verifies:
1. EnvironmentComplianceSuite compliance (dataclass, BaseRLEnvironment inheritance, obs_dim, action_dim).
2. Reset and step lifecycle contracts under pure JAX and jax.jit.
3. Preprocessing, postprocessing, and bounds clamping.
4. Custom MJCF XML loading.
5. Integration with RLEvaluator, MLPInterpreter, and ScalarOutput.
6. MalthusJAX optimization polarity contract (maximize=True -> lower-is-better negative fitness).
7. Population-level batch evaluation via jax.vmap.

DISAMBIGUATION COMPLIANCE:
- Mandatory alias: `from mujoco import mjx as mujoco_mjx`
- Entities prefixed: `MuJoCoMJXEnv`, `MuJoCoMJXState`, `mujoco_model`
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
from malthusjax.core.fitness.composable.environments import MuJoCoMJXEnv, MuJoCoMJXState
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.testing.compliance.environment_suite import EnvironmentComplianceSuite

CUSTOM_PENDULUM_MJCF = """
<mujoco model="simple_pendulum">
  <compiler inertiafromgeom="true"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1"/>
    <body pos="0 0 1" name="arm">
      <joint name="pin" type="hinge" axis="0 1 0"/>
      <geom type="cylinder" size="0.05 0.3" mass="1.0" pos="0 0 -0.3"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="pin" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


# =============================================================================
# 1. Compliance Suite
# =============================================================================


class TestMuJoCoMJXEnvCompliance(EnvironmentComplianceSuite):
    """Verifies MuJoCoMJXEnv satisfies MalthusJAX EnvironmentComplianceSuite."""

    @pytest.fixture
    def component(self) -> BaseEnvironment:
        return MuJoCoMJXEnv.create("cartpole", max_steps=100)


# =============================================================================
# 2. Lifecycle & Contracts
# =============================================================================


class TestMuJoCoMJXEnvLifecycle:
    def test_instantiation_cartpole(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=100)
        assert env.obs_dim == 4  # nq=2 (slider, hinge) + nv=2
        assert env.action_dim == 1  # 1 motor actuator
        assert env.max_steps == 100

    def test_reset_contract(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=50)
        key = jax.random.PRNGKey(123)
        obs, state = env.reset(key)

        assert isinstance(state, MuJoCoMJXState)
        assert obs.shape == (4,)
        assert jnp.all(jnp.isfinite(obs))
        assert state.step_count == 0
        assert not bool(state.done)
        assert jax.tree_util.tree_structure(state) is not None

    def test_step_contract(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=50)
        key = jax.random.PRNGKey(42)
        k_reset, k_step = jax.random.split(key)

        obs, state = env.reset(k_reset)
        action = jnp.array([0.5], dtype=jnp.float32)

        next_obs, next_state, reward, done, info = env.step(state, action, k_step)
        assert next_obs.shape == (4,)
        assert jnp.all(jnp.isfinite(next_obs))
        assert isinstance(next_state, MuJoCoMJXState)
        assert next_state.step_count == 1
        assert jnp.isfinite(reward)
        assert isinstance(done, jax.Array)

    def test_step_jit_compilation(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=50)
        key = jax.random.PRNGKey(99)
        k_reset, k_step = jax.random.split(key)

        jit_step = jax.jit(env.step)
        _, state = env.reset(k_reset)
        action = jnp.array([-0.25], dtype=jnp.float32)

        next_obs, next_state, reward, done, _ = jit_step(state, action, k_step)
        assert next_obs.shape == (4,)
        assert next_state.step_count == 1

    def test_preprocess_and_postprocess(self):
        env = MuJoCoMJXEnv.create("cartpole")
        raw_obs = jnp.ones((2, 2), dtype=jnp.float32)
        flat = env.preprocess_obs(raw_obs)
        assert flat.shape == (4,)

        logits = jnp.array([2.0, -3.0], dtype=jnp.float32)
        actions = env.postprocess_action(logits)
        assert jnp.all(actions >= -1.0) and jnp.all(actions <= 1.0)

    def test_custom_xml_string_loading(self):
        env = MuJoCoMJXEnv.create(xml_string=CUSTOM_PENDULUM_MJCF, max_steps=75)
        assert env.obs_dim == 2  # nq=1 + nv=1
        assert env.action_dim == 1  # 1 torque motor
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert obs.shape == (2,)


# =============================================================================
# 3. Composable Evaluator & Polarity Integration
# =============================================================================


class TestMuJoCoMJXComposableIntegration:
    def test_rlevaluator_single_rollout(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=50)
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(8,),
            activation="tanh",
        )
        output = ScalarOutput(maximize=True)
        evaluator = RLEvaluator(
            env=env,
            transform=IdentityTransform(),
            interpreter=interpreter,
            output=output,
            max_steps=50,
            num_eval_envs=2,
        )

        genome = RealGenome(values=jnp.zeros(interpreter.num_params))
        key = jax.random.PRNGKey(42)
        fitness, info = evaluator.evaluate(genome, genome, key)

        # Polarity contract: maximize=True must invert positive reward into negative fitness
        assert fitness.shape == ()
        assert jnp.isfinite(fitness)
        assert float(fitness) <= 0.0, "Polarity check failed: maximize=True must yield negative fitness!"

    def test_population_evaluation_vmap(self):
        env = MuJoCoMJXEnv.create("cartpole", max_steps=30)
        interpreter = MLPInterpreter(
            input_dim=env.obs_dim,
            output_dim=env.action_dim,
            hidden=(8,),
            activation="tanh",
        )
        output = ScalarOutput(maximize=True)
        evaluator = RLEvaluator(
            env=env,
            transform=IdentityTransform(),
            interpreter=interpreter,
            output=output,
            max_steps=30,
            num_eval_envs=1,
        )

        pop_size = 8
        key = jax.random.PRNGKey(101)
        k_pop, k_eval = jax.random.split(key)
        weights = jax.random.normal(k_pop, (pop_size, interpreter.num_params))
        config = RealGenomeConfig(shape=(interpreter.num_params,))
        pop = RealPopulation(
            genes=RealGenome(values=weights),
            fitness=jnp.full((pop_size,), jnp.nan),
            config=config,
        )

        evaluated_pop = evaluator.evaluate_population(pop, rng=k_eval)
        assert evaluated_pop.fitness.shape == (pop_size,)
        assert jnp.all(jnp.isfinite(evaluated_pop.fitness))
        assert jnp.all(evaluated_pop.fitness <= 0.0)
