# Memorandum: OmegaConf Integration for Composer

## Overview
This document outlines the strategic implementation plan for introducing `OmegaConf` into MalthusJAX's `Composer` API. The goal is to gain advanced configuration capabilities (variable interpolation, object instantiation, YAML support) while strictly maintaining backward compatibility with the existing TOML-based system and `OperatorCatalog`.

We are adopting a **"Lightweight Integration" (Path B)** approach, meaning we will leverage `OmegaConf` for parsing and interpolation, but we will **not** adopt the heavy Hydra framework (`@hydra.main`).

## 1. Core Objectives
1. **Enable Variable Interpolation**: Allow users to link hyperparameters mathematically (e.g., `${eval: ${num_evals} / ${pop_size}}`).
2. **Enable Custom Object Instantiation**: Allow users to define custom operators via `_target_` keys, bypassing the need for central registries.
3. **Dual Format Support**: Support both `.toml` and `.yaml`/`.yml` seamlessly.
4. **100% Backward Compatibility**: Ensure all existing TOML configurations and string-based operator specifications (e.g., `"tournament:num_selections=3"`) continue to work without modification.

---

## 2. Implementation Steps

### Phase 1: Dependency and Parser Upgrades
1. **Add Dependency:**
   - Add `omegaconf` and `pyyaml` to `pyproject.toml` dependencies.
2. **Update `config.py:load_experiment_config`:**
   - Refactor the loader to check file extensions (`.toml` vs `.yaml`).
   - Load the raw file into a Python dictionary (`tomllib.load` or `yaml.safe_load`).
   - Wrap the raw dictionary in `OmegaConf.create()`.
   - Call `OmegaConf.resolve(config)` to evaluate all `${...}` expressions before passing it to `Composer`.

### Phase 2: Object Instantiation Utility
1. **Create `config.py:instantiate`:**
   - Implement a lightweight, standalone `instantiate(config: DictConfig) -> Any` function.
   - It should extract the `_target_` key, dynamically import the module and class, and initialize the object using the remaining config kwargs.

### Phase 3: Composer API Adjustments
1. **Rename API Entrypoint (Optional but Recommended):**
   - Introduce `Composer.from_config()` as the primary entry point.
   - Maintain `Composer.from_toml()` as a deprecated alias that simply calls `from_config()` to guarantee legacy scripts do not break.
2. **Dual-Syntax Resolution in `quick_run`:**
   - Update operator resolution logic inside `Composer.quick_run` (and related factories) to check the type of the spec:
     - `if isinstance(spec, str)`: Route to the legacy `OperatorCatalog.build(spec)`.
     - `if isinstance(spec, (dict, DictConfig)) and "_target_" in spec`: Route to the new `instantiate(spec)`.

---

## 3. Example Usage After Migration

### Example 1: Pure Backward Compatibility (Legacy TOML)
```toml
# This continues to work perfectly!
[experiment.shared]
pop_size = 50
selection = "tournament:num_selections=3"
```

### Example 2: The New Way (YAML + Interpolation + _target_)
```yaml
experiment:
  shared:
    pop_size: 1024
    num_evals: 1000000
    # Dynamic computation!
    generations: ${eval: ${.num_evals} / ${.pop_size}}
    
    mutation:
      # Inject custom classes directly!
      _target_: malthusjax.operators.mutation.real.RealMutation
      rate: 0.1
```

## 4. Open Questions / Design Decisions
- **Deprecation Timeline:** If we rename `from_toml` to `from_config`, how long do we keep the alias before officially removing it?
- **Validation layer:** Should we eventually introduce `Pydantic` schemas for structured validation *after* OmegaConf resolves the variables? (This is a potential Phase 4).
