# Brax Implementation Checklist

- [ ] Create `src/malthusjax/core/fitness/rl/brax_evaluator.py`.
- [ ] Implement `BraxEvaluator` class extending `BaseEvaluator`.
- [ ] Implement `__init__` capturing `env_name`, `policy_module`, and `max_steps`.
- [ ] Implement `_rollout_episode` static/class method using `jax.lax.scan` adapted for Brax state.
- [ ] Implement `evaluate` utilizing the rollout loop.
- [ ] Implement `evaluate_population` using `jax.vmap`.
- [ ] Add unit tests in `tests/core/fitness/rl/test_brax_evaluator.py`.
- [ ] Test integration with `ant` or `halfcheetah` and a continuous MLP policy.
