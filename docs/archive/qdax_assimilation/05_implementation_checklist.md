# Implementation Checklist

This checklist defines the step-by-step engineering tasks required to assimilate QDax into MalthusJAX.

## Step 1: Base Interface Extensions
- [ ] Modify `src/malthusjax/core/fitness/base.py`
  - Add `evaluate_qd(self, genomes: PyTree) -> Tuple[jnp.ndarray, jnp.ndarray]` to `BaseEvaluator`.
  - Ensure legacy `evaluate` methods remain untouched for backward compatibility.
- [ ] Create `src/malthusjax/operators/emitters/base.py`
  - Define the `BaseEmitter` protocol (`init_state` and `emit`).
- [ ] Define `QDaxEngineState` inside `src/malthusjax/engine/state.py` or a new `qd_state.py`.
  - Must inherit from `BaseEngineState` and accept PyTrees for `repertoire` and `emitter_state`.

## Step 2: The QDax Adapter Wrappers
- [ ] Create `src/malthusjax/compat/qdax/evaluator.py`
  - Implement `QDaxEvaluatorAdapter(BaseEvaluator)` to wrap QDax brax/jumanji tasks.
- [ ] Create `src/malthusjax/compat/qdax/emitter.py`
  - Implement `QDaxEmitterAdapter(BaseEmitter)` to wrap QDax standard emitters (e.g. `MixingEmitter`).

## Step 3: Phase 1 (Level 1) Implementation
- [ ] Create `src/malthusjax/engine/qd_adapter_engine.py`
  - Implement `QDaxEngineAdapter(BaseEngine)`.
  - Route the `.run()` method directly to `MAPElites.scan_update()`.
- [ ] Register the components in the Composer catalog.
  - `@register_engine("qdax_adapter")`
  - `@register_evaluator("qdax_task")`
  - `@register_emitter("qdax_mixing_emitter")`
- [ ] **Validation**: Write a TOML experiment config and run `mjax run configs/test_qdax_adapter.toml` to ensure it executes without crashing.

## Step 4: Phase 2 (Level 2) Implementation
- [ ] Create `src/malthusjax/engine/native_qd_engine.py`
  - Implement `NativeQDEngine(BaseEngine)`.
  - Implement the `ask()` method to extract genotypes via the emitter.
  - Implement the `tell()` method to push evaluated genotypes into the repertoire.
  - Rewrite the `_scan_fn` loop to natively orchestrate `ask -> evaluate_qd -> tell`.
- [ ] **Validation**: Swap the engine string in the TOML config from `"qdax_adapter"` to `"native_qd_engine"` and ensure the exact same QD metric scores are achieved over 50 generations.
