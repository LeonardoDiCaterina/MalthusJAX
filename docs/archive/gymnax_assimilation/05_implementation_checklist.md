# Gymnax Implementation Checklist

- [ ] Create `src/malthusjax/core/fitness/rl/gymnax_evaluator.py`.
- [ ] Implement `GymnaxEvaluator` class extending `BaseEvaluator`.
- [ ] Implement `__init__` capturing `env_name`, `policy_module`, and `max_steps`.
- [ ] Implement `_rollout_episode` static/class method using `jax.lax.scan`.
- [ ] Implement `evaluate` utilizing the rollout loop.
- [ ] Implement `evaluate_population` using `jax.vmap`.
- [ ] Add unit tests in `tests/core/fitness/rl/test_gymnax_evaluator.py`.
- [ ] Test integration with `CartPole-v1` and a simple MLP policy.
