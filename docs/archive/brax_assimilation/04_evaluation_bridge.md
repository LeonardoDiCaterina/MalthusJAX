# Brax Evaluation Bridge

This document details the code structure for bridging MalthusJAX and Brax.

## The `lax.scan` Rollout
Brax's `step` function does not explicitly take an RNG key, simplifying the loop slightly.

```python
def rollout_episode(rng_input, policy_params):
    env_state = env.reset(rng_input)

    def step_fn(carry, _):
        env_state, cum_reward, done = carry
        
        # 1. Forward Pass (Policy)
        action = policy.apply(policy_params, env_state.obs)
        
        # 2. Environment Step
        next_state = env.step(env_state, action)
        
        # 3. Mask Reward
        reward = next_state.reward * (1.0 - done)
        new_cum_reward = cum_reward + reward
        new_done = jnp.logical_or(done, next_state.done)
        
        return (next_state, new_cum_reward, new_done), None

    # Initial carry
    carry_init = (env_state, 0.0, jnp.array(False, dtype=bool))
    
    # Run scan
    final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=max_steps)
    _, total_reward, _ = final_carry
    
    return total_reward
```

## Considerations
Brax observations and states are complex Pytrees. When using `jax.lax.scan` or `jax.vmap`, JAX seamlessly handles these Pytrees, but users should be cautious if attempting to manually unpack the `State` object inside the `step_fn`.
