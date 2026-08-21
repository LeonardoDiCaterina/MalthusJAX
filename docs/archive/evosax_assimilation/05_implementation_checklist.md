# Implementation Checklist (Evosax)

This checklist tracks the implementation of the Evosax Universal Adapter integration.

- [x] Analyze `evosax` API design and state management.
- [x] Design universal `@adapter` parameters for `evosax` (in `03_adapter_mapping.md`).
- [x] Create `EvosaxEngineAdapter` using the `@adapter` decorator in `src/malthusjax/composer/evosax_adapter.py`.
- [x] Validate execution against `EvalMode.NATIVE` using Evosax's `BBOB_Fitness`.
- [x] Ensure 80%+ test coverage for the integration logic.
