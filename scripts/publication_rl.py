"""Publication-only rollout instrumentation; optimizer implementations unchanged."""
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.evaluators import RLEvaluator


def all_finite(tree):
    leaves = [x for x in jax.tree_util.tree_leaves(tree)
              if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)]
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])) if leaves else jnp.array(True)


def dynamic_values(state):
    # Do not flag intentionally padded contact/constraint buffers as anomalies.
    pipeline = getattr(state, "pipeline_state", None)
    return [getattr(pipeline, name) for name in ("q", "qd") if hasattr(pipeline, name)]


@struct.dataclass
class InstrumentedRLEvaluator(RLEvaluator):
    """Retain padded scans but expose live-step masks and simulation anomalies."""

    def evaluate(self, raw_genome, compiled_genome, rng, **kwargs):
        def episode(key):
            reset_key, step_key = jax.random.split(key)
            obs, state = self.env.reset(reset_key)
            reset_invalid = ~all_finite((obs, dynamic_values(state)))

            def step(carry, _):
                state, obs, key, total, done = carry
                key, action_key = jax.random.split(key)
                logits = self.interpreter.apply(compiled_genome, self.env.preprocess_obs(obs))
                action = self.env.postprocess_action(logits, obs)
                next_obs, next_state, reward, next_done, _ = self.env.step(state, action, action_key)
                invalid = ~all_finite((action, reward, next_obs, dynamic_values(next_state)))
                live = ~done
                valid = live & ~invalid
                safe_reward = jnp.where(valid, jnp.nan_to_num(reward), 0.0)
                infos = self.output.observe_step(state, obs, action, safe_reward)
                infos = {} if infos is None else dict(infos)
                infos.update(valid_step=valid, live_step=live, invalid_live_step=live & invalid,
                             invalid_scan_step=invalid, termination=live & next_done)
                safe_obs = jax.tree_util.tree_map(
                    lambda x: jnp.nan_to_num(x) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact) else x,
                    next_obs)
                return (next_state, safe_obs, key, total + safe_reward, done | next_done | invalid), infos

            final, infos = jax.lax.scan(step, (state, obs, step_key, jnp.array(0.0), reset_invalid),
                                        None, length=getattr(self.env, "max_steps", self.max_steps))
            return final[3], final[0], infos, reset_invalid

        rewards, states, steps, bad_reset = jax.vmap(episode)(jax.random.split(rng, self.num_eval_envs))
        fitness, info = self.output.process_rl(raw_genome, jnp.mean(rewards), states, steps)
        info = dict(info)
        info.update(rollout_live_steps=jnp.sum(steps["live_step"]),
                    rollout_valid_steps=jnp.sum(steps["valid_step"]),
                    rollout_invalid_live_steps=jnp.sum(steps["invalid_live_step"]),
                    rollout_invalid_scan_steps=jnp.sum(steps["invalid_scan_step"]),
                    rollout_terminations=jnp.sum(steps["termination"]),
                    rollout_invalid_resets=jnp.sum(bad_reset))
        return fitness, info
