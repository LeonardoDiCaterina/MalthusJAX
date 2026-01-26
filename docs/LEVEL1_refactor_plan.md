# Level 1 (Core) Refactor Plan ✅🔧

**Purpose:** Define a concrete, step-by-step plan to refactor the Level 1 (core) code in `src/malthusjax/core/` to be JAX-first, well-tested, typed, and fully linted. This plan focuses exclusively on Level 1 (core) and prepares the codebase to be safely used by higher levels.

---

## Goals 🎯
- **Correctness:** Maintain or improve existing behavior and benchmarks.
- **JAX-first API:** Ensure Level 1 follows the NEW JAX patterns (e.g., `@struct.dataclass`, `__call__` operator factories, vmap-ready, pure functions).
- **Test coverage:** Achieve >= 80% pytest coverage for Level 1 modules.
- **Quality gates:** Zero lint errors (Ruff), and no mypy type errors for Level 1.
- **CI enforcement:** Add CI jobs to run tests, lint, and type checks on PRs touching core.

---

## Scope ✳️
**Includes:**
- `src/malthusjax/core/` modules, their unit tests and docs
- Tests for random key utilities and PyTree behaviors used by core
- Linting and type-check configuration relevant to core

**Out of scope:** operators, engines, examples, and Level 2+ refactors (handled in later phases)

---

## Deliverables 📦
- `docs/LEVEL1_refactor_plan.md` (this file)
- Updated `pyproject.toml` / Ruff config adjustments (if needed)
- New/updated tests under `tests/core/`
- CI workflow(s): `ci/core-tests.yml` (run tests, coverage, lint, mypy)
- PR checklist template for Level 1 changes

---

## High-Level Milestones & Tasks (step-by-step) 🛠️

1. Baseline & Inventory (2 days) 💡
   - Run tests and coverage for core: `pytest --cov=src/malthusjax/core tests/core/`.
   - Generate a list of core modules lacking tests and record current coverage per module.
   - Run Ruff and mypy and record current errors/warnings affecting core.

2. Linting & Type Hygiene (3 days) ⚠️
   - Fix Ruff issues in `src/malthusjax/core/` until `ruff check` is clean for those files.
   - Add/adjust `pyproject.toml` config to set allowed exceptions for now (if any), but aim to remove them.
   - Run `mypy` on core modules and fix type errors and missing annotations.

3. Test Coverage Expansion (1–2 weeks) ✅
   - Add unit tests for every public function/class in core (see test checklist below).
   - Focus on edge cases, JIT-compatibility scenarios, and vectorized behavior using `jax.vmap`.
   - Use fixtures for deterministic random keys (see `tests/conftest.py` patterns).

4. API Refactor & Conformance (2–3 weeks) 🧩
   - Convert core dataclasses to `@struct.dataclass` where appropriate, with explicit factory methods like `random_init`.
   - Remove or refactor any `__post_init__` usage (moved to factory/assertion outside jit boundary).
   - Ensure PyTree registration for config dataclasses.
   - Ensure operator signatures follow the NEW canonical order (key first, batch-first outputs).

5. Vectorization & JIT Tests (3–5 days) ⚡
   - Add tests verifying `jax.jit` and `jax.vmap` behaviors for initialization and evaluation.
   - Add tests for RNG splitting behavior (auto-splitting when using `random_key` property), following project pattern.

6. CI & Automation (2 days) 🔁
   - Add CI job `core-tests` that runs `pytest -k core`, coverage, `ruff check src/malthusjax/core/`, and `mypy src/malthusjax/core/`.
   - Fail PRs if coverage falls below target or lint/type checks fail.

7. Documentation & Examples (3 days) 📚
   - Add short docs and examples that demonstrate correct usage patterns for Level 1 classes (e.g., `random_init`, `__call__` operators, vectorizing with `vmap`).

8. Final Sweep & Merge Strategy (1 week) 🧪
   - Break work into small, reviewable PRs (module-by-module).
   - Add PR template items: tests added/updated, linting fixed, coverage impact.
   - Use feature branches and squash/merge strategy.

---

## Test Checklist (must be covered per module) ✅
- [ ] Initialization correctness: shapes, defaults, and random_init determinism
- [ ] PyTree registration and dataclass immutability
- [ ] RNG key management & auto-splitting behavior
- [ ] JIT & vmap compatibility (no Python-side checks that break tracing)
- [ ] Behavioral equivalence before/after refactor for public APIs
- [ ] Edge cases (empty inputs, broadcast behavior)
- [ ] Exception messages and validation moved outside JIT

---

## Linting & Type Checklist ⚖️
- [ ] `ruff check src/malthusjax/core/` returns zero errors
- [ ] `mypy --strict` passes for core modules (address missing types incrementally)
- [ ] Add or update `pyproject.toml` to document core-specific strictness levels

---

## CI Changes 🧾
- Add `ci/core-tests.yml` to run on PRs touching `src/malthusjax/core/**`:
  - Steps: install dependencies, run `ruff`, `mypy`, `pytest --cov=... --cov-fail-under=80` limited to core tests, upload coverage report.
  - Optionally add caching for pip/poetry deps.

---

## Acceptance Criteria ✅
- Per-module coverage >= 80% (or agreed threshold) for core
- No Ruff or mypy errors for files under `src/malthusjax/core/`
- Tests demonstrating JAX/jit/vmap compatibility
- CI job passing for core checks and gating merges

---

## Risks & Mitigations ⚠️
- Risk: Refactor breaks higher-level expectations. Mitigation: keep behavior tests and use small incremental PRs.
- Risk: Long-running CI. Mitigation: Create focused CI job for core that runs in parallel to full-suite jobs.

---

## Timeline & Priority (example 6-week plan) ⏳
- Week 1: Baseline, minor lint/type fixes, start tests
- Week 2–3: Add tests broadly, refactor the lowest-risk core modules
- Week 4: Complete harder refactors (dataclass conversion, RNG behavior)
- Week 5: CI integration, docs, examples
- Week 6: Final sweep, merge queued PRs

---

## Next Actions (immediate) ▶️
1. Create a GitHub issue: "Level 1 (core) refactor — plan and roadmap" and attach this plan.
2. Run baseline commands and capture outputs: `pytest --maxfail=1 --disable-warnings -q` and `ruff check src/malthusjax/core/` and `mypy src/malthusjax/core/`.
3. Start with an inventory PR: add per-module test stubs in `tests/core/` and a CI job skeleton.

---

If you'd like, I can: 
- Create the initial `tests/core/` skeleton and baseline CI workflow, or
- Run the baseline commands and produce the coverage/lint/mypy reports to prioritize next modules.

---

*Author:* GitHub Copilot
*Date:* 2026-01-25
