# PR2 — Benchmark Runner & IO: Guide and Checklist

Status: Draft • Owner: you

## Summary
Implement a shallow, testable BenchmarkRunner and IO helpers, wire `Composer.quick_run` to produce canonical ExperimentResult artifacts (JSON summary + combined history CSV), and add a small, fast integration smoke test. Keep this PR focused and side-effect-free (no heavy JAX work). This is PR2 in the Composer/BENCHMARKING effort.

---

## Goals
- Add datamodels for run & experiment results
- Provide IO helpers to write `summary.json` and `histories_combined.csv` (with `seed` column)
- Implement `BenchmarkRunner` skeleton and stubbed engine contract
- Add `Composer.quick_run` glue and a CLI wrapper
- Include unit + small integration tests that run quickly in CI

---

## Files to add or modify

### New
- `src/malthusjax/benchmarking/results.py` — dataclasses:
  - `RunResult`, `ExperimentResult` and `to_dict` / `to_json` helpers
- `src/malthusjax/benchmarking/io.py` — helpers:
  - `write_summary_json(experiment_result, path)`
  - `write_history_csv(history_df, path)`
  - `ensure_seed_folder(output_dir, seed)`
- `src/malthusjax/benchmarking/runner.py` — `BenchmarkRunner`:
  - `run(seeds: Sequence[int], profiler=False)` → `ExperimentResult`
  - Stubbed calls to `engine.run_once(key)` for now

### Integrations
- `src/malthusjax/composer/composer.py` — add `quick_run(...)` that builds the pipeline, constructs `BenchmarkRunner`, and runs
- `src/malthusjax/benchmarking/cli.py` — thin CLI entrypoint to call `Composer.quick_run`

### Tests
- `tests/benchmarking/test_results.py` — dataclass and JSON round-trip
- `tests/benchmarking/test_io.py` — write/read validation using `tmp_path`
- `tests/benchmarking/test_runner.py` — smoke test using a deterministic stub engine
- `tests/composer/test_quick_run.py` — end-to-end quick-run smoke test

---

## Commit plan (small, review-friendly)
1. Commit A — Add `results.py` + tests
2. Commit B — Add `io.py` + tests
3. Commit C — Add `runner.py` (stub engine) + tests
4. Commit D — Wire `Composer.quick_run` + `tests/composer/test_quick_run.py`
5. Commit E — CLI + example notebook update + docs

Write focused commit messages (one responsibility per commit).

---

## Engine stub contract (for tests)
- `engine.run_once(key) -> dict` with keys:
  - `history`: list[dict] with per-generation stats
  - `summary`: final summary dict (e.g., `{best_fitness, best_gen}`)
  - `timings`: optional timing metrics

This keeps `BenchmarkRunner` engine-agnostic and easy to test.

---

## Example `BenchmarkRunner.run` behavior
1. Normalize seeds (`seeds` or `seed+repeats`).
2. For each seed: derive subkey, call `engine.run_once()`, collect history and summary.
3. Persist per-seed history and append to a combined CSV with `seed` column.
4. Build `ExperimentResult`:
   - `canonical_summary` is the summary of the first seed (for backward compatibility)
   - `aggregated_summary` includes mean/median of best fitness across seeds
5. Write `summary.json` and return the `ExperimentResult`.

---

## Tests & Acceptance Criteria
- Unit tests for `RunResult`/`ExperimentResult` and IO functions pass.
- Integration smoke (`test_runner` and `test_quick_run`) pass quickly and deterministically.
- `Composer.quick_run` returns an `ExperimentResult` with per-seed entries and writes artifacts to `output_dir`.
- No heavy JAX activity at import time.

---

## Commands to run locally
- Run benchmarking tests:
  ```bash
  PYTEST_ADDOPTS="" pytest -q tests/benchmarking
  ```
- Run composer quick-run test:
  ```bash
  PYTEST_ADDOPTS="" pytest -q tests/composer/test_quick_run.py
  ```
- Lint/type checks for new code:
  ```bash
  ruff check src/malthusjax/benchmarking tests/benchmarking
  mypy --ignore-missing-imports --python-version 3.12 src/malthusjax/benchmarking src/malthusjax/composer
  ```

---

## CI notes
- Add a CI job to run the new tests and type/lint checks (fast configs, small test sizes).
- Keep longer-running benchmarks in separate workflows.

---

## Risks & mitigations
- **JAX tracing at composition time**: only call engine operations inside `run()`, not during composition.
- **Long tests**: enforce small default sizes for tests and CI smoke runs.
- **Artifact collisions**: include `seed` or timestamp in filenames to avoid overwriting.

---

## Reviewer checklist
- Tests are deterministic and fast.
- APIs are documented and typed.
- No device allocation or JAX `jit` at import time.
- Artifacts are persisted under the expected layout.

---

## Next steps
- Confirm you want the scaffolding and I will create Commit A (add `results.py` + tests) and push to `feat/benchmarking-runner` branch for your review.

---

If you want me to scaffold Commit A now, say **scaffold Commit A** and I'll add the files and tests to a new branch and open PR2 draft for you.