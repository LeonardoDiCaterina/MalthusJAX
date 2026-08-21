# Gymnax Evaluation Bridge

This document describes the specific implementation of the rollout loop linking MalthusJAX populations to Gymnax environments.

## The `lax.scan` Rollout
A typical RL rollout in JAX requires `lax.scan` to avoid Python-level loops and allow JIT compilation.

```python
def rollout_episode(rng_input, policy_params):
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, env_state = env.reset(rng_reset, env_params)

    def step_fn(carry, _):
        env_state, obs, rng, cum_reward, done = carry
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        
        # 1. Forward Pass (Policy)
        action = policy.apply(policy_params, obs)
        
        # 2. Environment Step
        next_obs, next_state, reward, next_done, info = env.step(rng_step, env_state, action, env_params)
        
        # 3. Mask Reward
        reward = reward * (1.0 - done)
        new_cum_reward = cum_reward + reward
        new_done = jnp.logical_or(done, next_done)
        
        return (next_state, next_obs, rng, new_cum_reward, new_done), None

    # Initial carry
    carry_init = (env_state, obs, rng_episode, 0.0, False)
    
    # Run scan
    final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=max_steps)
    _, _, _, total_reward, _ = final_carry
    
    return total_reward
```

## Batching and VMAP
The fitness of a given individual is typically the mean reward over N environment seeds.

```python
# Vmap over seeds for a single individual
vmap_seeds = jax.vmap(rollout_episode, in_axes=(0, None))
# Vmap over the population of individuals
vmap_pop = jax.vmap(vmap_seeds, in_axes=(None, 0))
```

This bridge allows MalthusJAX to natively support multi-objective fitness structures if needed (e.g., returning episode length and reward).
