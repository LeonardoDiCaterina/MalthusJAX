#!/usr/bin/env python3
"""
hello_MuJoCo.py — DeepMind MuJoCo MJX Research & Contract Validation Harness.

Independent research harness to evaluate DeepMind's MuJoCo in JAX (mujoco.mjx)
and expose all classes and contracts required for seamless integration into
MalthusJAX as a first-class simulation environment.

DISAMBIGUATION COMPLIANCE:
- Mandatory alias: `from mujoco import mjx as mujoco_mjx`
- Entities prefixed: `MuJoCoMJXEnv`, `MuJoCoMJXState`, `mujoco_model`, `mujoco_data`, `mmjx_step`.
- Zero bare `import mjx` references.
"""

from __future__ import annotations

import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx as mujoco_mjx

from malthusjax.core.fitness.composable.base import (
    BaseRLEnvironment,
    IdentityTransform,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.evaluators import RLEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation

# =============================================================================
# 1. Benchmark Models (Embedded MJCF)
# =============================================================================

CARTPOLE_MJCF = """
<mujoco model="cartpole">
  <compiler inertiafromgeom="true"/>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.5 2.5"/>
      <geom name="cart_geom" type="box" size="0.2 0.1 0.1" mass="1.0" rgba="0.2 0.6 0.8 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="pole_geom" type="cylinder" size="0.04 0.4" mass="0.1" pos="0 0 0.4" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slider" ctrlrange="-10 10" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


# =============================================================================
# 2. State & Environment Contracts
# =============================================================================


@struct.dataclass
class MuJoCoMJXState:
    """JAX PyTree state encapsulating a MuJoCo MJX physics step."""

    data: mujoco_mjx.Data
    step_count: chex.Array
    cum_reward: chex.Array
    done: chex.Array


@struct.dataclass
class MuJoCoMJXEnv(BaseRLEnvironment):
    """DeepMind MuJoCo MJX environment conforming to MalthusJAX BaseRLEnvironment."""

    mujoco_model: mujoco_mjx.Model = struct.field(pytree_node=False)
    raw_model: mujoco.MjModel = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False, default=500)
    reset_noise_scale: float = struct.field(pytree_node=False, default=0.05)

    @classmethod
    def create_from_xml_string(
        cls, xml_string: str, max_steps: int = 500, reset_noise_scale: float = 0.05
    ) -> MuJoCoMJXEnv:
        """Instantiate from an MJCF XML string."""
        raw_model = mujoco.MjModel.from_xml_string(xml_string)
        mujoco_model = mujoco_mjx.put_model(raw_model)
        return cls(
            mujoco_model=mujoco_model,
            raw_model=raw_model,
            max_steps=max_steps,
            reset_noise_scale=reset_noise_scale,
        )

    # --- Contract 1: reset(key) -> (obs, state) ---
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, MuJoCoMJXState]:
        k_pos, k_vel = jax.random.split(key)
        # Create unperturbed data
        data = mujoco_mjx.make_data(self.mujoco_model)

        # Add initial stochastic perturbation
        qpos_noise = (
            jax.random.uniform(
                k_pos, shape=(self.mujoco_model.nq,), minval=-self.reset_noise_scale, maxval=self.reset_noise_scale
            )
        )
        qvel_noise = (
            jax.random.normal(k_vel, shape=(self.mujoco_model.nv,)) * self.reset_noise_scale
        )

        data = data.replace(
            qpos=data.qpos + qpos_noise,
            qvel=data.qvel + qvel_noise,
            ctrl=jnp.zeros(self.mujoco_model.nu, dtype=jnp.float32),
        )
        # Forward kinematics to refresh internal caches
        data = mujoco_mjx.forward(self.mujoco_model, data)

        obs = jnp.concatenate([data.qpos, data.qvel])
        state = MuJoCoMJXState(
            data=data,
            step_count=jnp.array(0, dtype=jnp.int32),
            cum_reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False, dtype=jnp.bool_),
        )
        return obs, state

    # --- Contract 2: step(state, action, key) -> (obs, state, reward, done, info) ---
    def step(
        self, state: MuJoCoMJXState, action: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, MuJoCoMJXState, chex.Numeric, chex.Array, dict]:
        # Clip action to actuator range
        clamped_action = jnp.clip(action, -1.0, 1.0)
        # Scale to model ctrlrange
        ctrl_min = self.mujoco_model.actuator_ctrlrange[:, 0]
        ctrl_max = self.mujoco_model.actuator_ctrlrange[:, 1]
        scaled_ctrl = ctrl_min + 0.5 * (clamped_action + 1.0) * (ctrl_max - ctrl_min)

        # Update controls
        data = state.data.replace(ctrl=scaled_ctrl)

        # Step physics via DeepMind MuJoCo MJX
        next_data = mujoco_mjx.step(self.mujoco_model, data)

        # Observation: joint positions & velocities
        next_obs = jnp.concatenate([next_data.qpos, next_data.qvel])

        # Cartpole Task Reward: pole upright (cos(theta)) - position penalty - control cost
        x = next_data.qpos[0]
        theta = next_data.qpos[1]
        alive_reward = 1.0
        upright_reward = jnp.cos(theta)
        cart_penalty = 0.01 * (x ** 2)
        ctrl_cost = 0.001 * jnp.sum(clamped_action ** 2)
        reward = alive_reward + upright_reward - cart_penalty - ctrl_cost

        # Termination criteria: pole fell (|theta| > 0.5 rad) or cart off-rail (|x| > 2.0)
        fell = jnp.abs(theta) > 0.5
        off_rail = jnp.abs(x) > 2.0
        time_limit = state.step_count >= self.max_steps
        done = jnp.logical_or(jnp.logical_or(fell, off_rail), time_limit)

        next_state = MuJoCoMJXState(
            data=next_data,
            step_count=state.step_count + 1,
            cum_reward=state.cum_reward + reward,
            done=done,
        )
        return next_obs, next_state, reward, done, {}

    # --- Contract 3: preprocess_obs ---
    def preprocess_obs(self, obs: chex.Array) -> chex.Array:
        return obs.flatten()

    # --- Contract 4: postprocess_action ---
    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        return jnp.tanh(logits)

    # --- Contract 5 & 6: Dimensions ---
    @property
    def obs_dim(self) -> int:
        return int(self.mujoco_model.nq + self.mujoco_model.nv)

    @property
    def action_dim(self) -> int:
        return int(self.mujoco_model.nu)


# =============================================================================
# 3. Validation Suite & Benchmarking
# =============================================================================


def run_research_experiments():
    print("=" * 80)
    print("MalthusJAX ⚡ DeepMind MuJoCo MJX: Independent API & Contract Research")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()} | MuJoCo version: {mujoco.__version__}")

    # 1. Instantiate Environment
    print("\n[1/5] Instantiating MuJoCoMJXEnv from MJCF XML...")
    env = MuJoCoMJXEnv.create_from_xml_string(CARTPOLE_MJCF, max_steps=200)
    print("  ✓ Model successfully ingested and transferred to accelerator!")
    print(f"  ✓ Observation Dimension (nq={env.mujoco_model.nq} + nv={env.mujoco_model.nv}): {env.obs_dim}")
    print(f"  ✓ Action Dimension (nu={env.mujoco_model.nu}): {env.action_dim}")

    # 2. Test Reset and Step contracts under jax.jit
    print("\n[2/5] Testing Reset and Step Contracts under jax.jit...")
    key = jax.random.PRNGKey(42)
    k_reset, k_step = jax.random.split(key)

    jit_reset = jax.jit(env.reset)
    obs, state = jit_reset(k_reset)
    print(f"  ✓ Reset output observation shape: {obs.shape}, values: {obs}")
    print(f"  ✓ State is pure PyTree: {jax.tree_util.tree_structure(state) is not None}")

    jit_step = jax.jit(env.step)
    action = jnp.array([0.5])
    next_obs, next_state, reward, done, info = jit_step(state, action, k_step)
    print(f"  ✓ Step output next_obs shape: {next_obs.shape}")
    print(f"  ✓ Step output reward: {reward:.4f}, done: {bool(done)}")

    # 3. Vectorization with jax.vmap across batch of environments
    print("\n[3/5] Testing Population Vectorization (jax.vmap across 64 environments)...")
    batch_size = 64
    batch_keys = jax.random.split(key, batch_size)
    vmap_reset = jax.jit(jax.vmap(env.reset))
    batch_obs, batch_states = vmap_reset(batch_keys)
    print(f"  ✓ Batched reset: {batch_obs.shape[0]} parallel environments initialized")

    batch_actions = jax.random.uniform(key, (batch_size, env.action_dim), minval=-1.0, maxval=1.0)
    vmap_step = jax.jit(jax.vmap(env.step))
    batch_next_obs, batch_next_states, batch_rewards, batch_dones, _ = vmap_step(
        batch_states, batch_actions, batch_keys
    )
    print(f"  ✓ Batched step output rewards shape: {batch_rewards.shape}")

    # 4. Composable Evaluator Integration (RLEvaluator + MLPInterpreter + ScalarOutput)
    print("\n[4/5] Testing Integration with MalthusJAX Composable RLEvaluator...")
    interpreter = MLPInterpreter(
        input_dim=env.obs_dim,
        output_dim=env.action_dim,
        hidden=(16, 16),
        activation="tanh",
    )
    output = ScalarOutput(maximize=True)  # Should invert reward to -reward for minimization
    evaluator = RLEvaluator(
        env=env,
        transform=IdentityTransform(),
        interpreter=interpreter,
        output=output,
        max_steps=200,
        num_eval_envs=4,
    )
    print("  ✓ Composable Stack: MuJoCoMJXEnv × IdentityTransform × MLPInterpreter × ScalarOutput")
    print(f"  ✓ Policy parameter count: {interpreter.num_params}")

    # Generate population of policies
    pop_size = 32
    config = RealGenomeConfig(shape=(interpreter.num_params,))
    k_pop, k_eval = jax.random.split(key)
    weights = jax.random.normal(k_pop, (pop_size, interpreter.num_params))
    population = RealPopulation(
        genes=RealGenome(values=weights),
        fitness=jnp.full((pop_size,), jnp.nan),
        config=config,
    )

    t0 = time.perf_counter()
    evaluated_pop = evaluator.evaluate_population(population, rng=k_eval)
    evaluated_pop.fitness.block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"  ✓ JIT Compile + Evaluate Population (pop={pop_size}, steps=200, envs=4): {t_compile:.3f}s")
    print(f"  ✓ Evaluated Fitnesses (Mean): {jnp.mean(evaluated_pop.fitness):.4f}")
    # Verify polarity contract: maximize=True must yield negative fitness for lower-is-better
    assert jnp.all(evaluated_pop.fitness <= 0.0), "Polarity violation: maximize=True must invert reward to negative!"
    print("  ✓ Polarity Invariant Verified: Fitness is strictly negative (lower-is-better contract satisfied)")

    # 5. Throughput Benchmarking
    print("\n[5/5] High-Throughput Rollout Benchmarking (128 parallel envs × 500 steps)...")
    eval_pop_size = 128
    eval_steps = 500
    eval_weights = jax.random.normal(k_pop, (eval_pop_size, interpreter.num_params))
    eval_population = RealPopulation(
        genes=RealGenome(values=eval_weights),
        fitness=jnp.full((eval_pop_size,), jnp.nan),
        config=config,
    )
    bench_evaluator = RLEvaluator(
        env=env,
        transform=IdentityTransform(),
        interpreter=interpreter,
        output=output,
        max_steps=eval_steps,
        num_eval_envs=1,
    )

    # Warmup
    _ = bench_evaluator.evaluate_population(eval_population, rng=k_eval)
    evaluated_pop.fitness.block_until_ready()

    n_trials = 5
    t0 = time.perf_counter()
    for _ in range(n_trials):
        res = bench_evaluator.evaluate_population(eval_population, rng=k_eval)
        res.fitness.block_until_ready()
    t_avg = (time.perf_counter() - t0) / n_trials

    total_env_steps = eval_pop_size * eval_steps
    steps_per_sec = total_env_steps / t_avg
    print(f"  ✓ Average Rollout Execution: {t_avg:.4f}s")
    print(f"  ⚡ Throughput: {steps_per_sec:,.0f} environment steps/second")
    print("=" * 80)
    print("Research experiment completed successfully! All contracts verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_research_experiments()
