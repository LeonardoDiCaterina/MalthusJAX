# Jumanji Evaluation Bridge

Code structure for bridging MalthusJAX and Jumanji.

## The `lax.scan` Rollout

```python
def rollout_episode(rng_input, policy_params):
    rng_reset, rng_episode = jax.random.split(rng_input)
    env_state, timestep = env.reset(rng_reset)

    def step_fn(carry, _):
        env_state, timestep, cum_reward, done = carry
        
        # 1. Forward Pass (Policy)
        # Note: the policy must handle the action_mask internally
        action = policy.apply(policy_params, timestep.observation)
        
        # 2. Environment Step
        next_state, next_timestep = env.step(env_state, action)
        
        # 3. Mask Reward
        reward = next_timestep.reward * (1.0 - done)
        new_cum_reward = cum_reward + reward
        new_done = jnp.logical_or(done, next_timestep.last())
        
        return (next_state, next_timestep, new_cum_reward, new_done), None

    # Initial carry
    carry_init = (env_state, timestep, 0.0, jnp.array(False, dtype=bool))
    
    # Run scan
    final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=max_steps)
    _, _, total_reward, _ = final_carry
    
    return total_reward
```

## Considerations
Jumanji environments often require highly specialized policy architectures to handle varying graph structures or sequence lengths. The `JumanjiEvaluator` should ideally be flexible enough to accept these specialized Flax modules.
