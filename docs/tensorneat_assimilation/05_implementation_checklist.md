# Implementation Checklist (TensorNEAT)

This checklist tracks the implementation of the TensorNEAT Universal Adapter integration.

- [x] Analyze `tensorneat` API design and state management.
- [x] Design universal `@adapter` parameters for `tensorneat` (in `03_adapter_mapping.md`).
- [ ] Implement `transform` support in the `@adapter` logic (since `tensorneat` requires transforming genotypes before evaluation).
- [ ] Implement manual `metrics_mapping` parsing in the `@adapter` scan loop (computing max fitness if `metrics` dictionary is not returned by `tell()`).
- [ ] Create `TensorNEATEngineAdapter` using the `@adapter` decorator in `src/malthusjax/composer/tensorneat_adapter.py`.
- [ ] Validate execution against `EvalMode.NATIVE` using TensorNEAT's standard evaluate problems.
- [ ] Ensure 80%+ test coverage for the integration logic.
