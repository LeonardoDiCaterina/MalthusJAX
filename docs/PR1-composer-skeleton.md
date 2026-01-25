# PR1 — Composer skeleton: tasks & checklist

Status: Draft • Owner: you

## Overview
Small, focused PR to add the `malthusjax.composer` skeleton with tests and a tiny quickstart example. Goal: provide a safe, side-effect-free foundation for composer and to enable PR2 (runner + IO) later.

Branch: `feat/composer-skeleton`
PR title: `feat(composer): add Registry, Node, Pipeline skeleton + quickstart tests`

## What to add (files)
- src/malthusjax/composer/__init__.py
  - re-export `Composer`, `Registry`, `Node`, `Pipeline`, `load_config`

- src/malthusjax/composer/registry.py
  - `class Registry`
    - methods: `register(name, factory, override: bool = True)`, `get(name)`, `list()`
    - store mapping and simple thread-safety

- src/malthusjax/composer/node.py
  - `@dataclass` Node(id: str, type: str, params: Dict[str, Any])
  - `build(self, key, registry, inputs=None)` calls factory

- src/malthusjax/composer/pipeline.py
  - `@dataclass` Pipeline(name: str, nodes: List[Node], wiring: Dict[str, Sequence[str]] = {})
  - `validate(registry)` checks missing nodes and wiring errors
  - `build(master_key, registry)` returns mapping id->object

- src/malthusjax/composer/config.py
  - TOML loader: `load_config(path, pipeline_name)` → `PipelineConfig` + normalized `seeds` list

- src/malthusjax/composer/presets.py
  - Opinionated `quick_demo` and `standard_ga` presets (small defaults)

- examples/quickstart.toml (small preset used by tests)

- tests/composer/test_registry.py
- tests/composer/test_node.py
- tests/composer/test_pipeline.py
- tests/composer/test_config.py
- tests/fixtures/quick_demo/ (tiny fixture config)

## Commit breakdown (one commit per logical group)
1. Add package exports & `__init__` (no logic)
2. Add `registry.py` + tests
3. Add `node.py` + tests
4. Add `pipeline.py` + tests
5. Add `config.py` + `presets.py` + tests (seed normalization tests)
6. Add `examples/quickstart.toml` and fixture
7. Add docs snippet (short README or docs/ note)

Keep commits small and self-contained; test after each commit.

## Unit tests & assertions
- Registry
  - can register & retrieve factory placeholders
  - test override semantics
- Node
  - Node serializes/deserializes
  - `build()` calls registry factory with correct params and key-splitting behavior
- Pipeline
  - `validate()` checks missing nodes and wiring errors
  - `build()` returns mapping id->object when registry factories are stubs
- Config loader
  - loads TOML, normalizes `seed` + `repeats` → `seeds: List[int]`
  - apply presets when fields missing
  - helpful error messages for invalid forms

## Acceptance criteria (PR checks)
- All tests pass locally and in CI
- `pip install -e .` works and imports `malthusjax.composer` without side effects
- `Composer.load_config("examples/quickstart.toml", "quick_demo").compose()` produces a `Pipeline` object
- `Composer.quick_run(...)` stub exists and returns a minimal `ExperimentResult` dataclass (no file IO in PR1)
- Documentation: short [`docs`](docs ) paragraph or README snippet describing `Composer` quick usage

## Linting, typing, docs
- Add type hints to public APIs (`Registry`, `Node`, `Pipeline`, config)
- Run `ruff` / `pre-commit` formatting before opening PR

## Reviewer checklist
- API ergonomics: method names intuitive
- Tests comprehensive for loader/validation and edge cases
- No JAX side effects on import or build
- Presets are small and CI-friendly

## Time estimate
- Scaffolding + tests: 3–6 hours
- Extra time for review and minor revisions: 1–2 hours

## Helpful snippets (to reuse in tests)
- Minimal factory stub for tests:

```py
def dummy_factory(key, params, inputs=None):
    return {"params": params, "inputs": inputs}
```

---

If you'd like, I can also scaffold the repository changes (create the files & tests) and leave TODOs inside them for you to implement; or I can just leave this file and you can start coding. Which do you prefer?