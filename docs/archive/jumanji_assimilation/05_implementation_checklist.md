# Jumanji Implementation Checklist

- [ ] Create `src/malthusjax/core/fitness/rl/jumanji_evaluator.py`.
- [ ] Implement `JumanjiEvaluator` class extending `BaseEvaluator`.
- [ ] Implement `__init__` capturing `env_name`, `policy_module`, and `max_steps`.
- [ ] Implement `_rollout_episode` static/class method using `jax.lax.scan` managing both `state` and `timestep`.
- [ ] Implement `evaluate` utilizing the rollout loop.
- [ ] Implement `evaluate_population` using `jax.vmap`.
- [ ] Add unit tests in `tests/core/fitness/rl/test_jumanji_evaluator.py`.
- [ ] Test integration with `Snake-v1` or `BinPack-v1` and a policy that supports masking.
