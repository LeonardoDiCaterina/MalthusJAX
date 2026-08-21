# Implementation Checklist (Kozax)

This checklist tracks the implementation of the Kozax Universal Adapter integration.

- [x] Analyze `kozax` API design and state management.
- [x] Design universal `@adapter` parameters for `kozax` (in `03_adapter_mapping.md`).
- [ ] Implement Monolithic `"step"` function handling in the `@adapter` logic (routing `state, fitness, key` directly to a step function when `ask` and `tell` are missing).
- [ ] Implement support for `metrics_mapping={"best_fitness": "min_fitness"}` to natively track minimization problems during the loop.
- [ ] Create `KozaxEngineAdapter` using the `@adapter` decorator in `src/malthusjax/composer/kozax_adapter.py`.
- [ ] Validate execution against `EvalMode.NATIVE` using Kozax's standard evaluate problems (e.g., Symbolic Regression).
- [ ] Ensure 80%+ test coverage for the integration logic.
