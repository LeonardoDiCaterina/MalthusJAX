# PT1 — exact prepared execution

Implemented on `codex/audit-remediation`. This closes the executable-reuse and preparation-failure portions of T01/T02 for native genetic and MAP-Elites execution. Barrier consistency and adoption by other adapters remain PT2/PT3 work.

- Kernel arguments are normalized before the bounded factory-cache lookup, so omitted/explicit defaults and keyword order select the same JIT function.
- `AbstractEngine.prepare_run()` retains an explicitly compiled executable, bound to the engine and effective logging/history options. `run(prepared_run=...)` invokes that compiled object directly. Incompatible configuration or input signatures fail instead of silently retracing.
- Both native benchmarking adapters prepare before starting execution timing. The MAP-Elites warmup exception suppression is removed. Genetic execution still passes through its override, preserving best-genome finalization.
- Preparation does not execute throwaway generations or mutate the measured seed/state. Compiled input donation remains in effect; regression reference runs use independent copies.

Regression evidence:

1. Before implementation, the nine initial prepared-run tests failed, including distinct default/explicit cache requests, measured tracing, and swallowed preparation failure (`/tmp/pt1-red.txt` during implementation).
2. `python3 -m pytest tests/engine/test_prepared_run.py tests/engine/test_base_abstract.py tests/test_timing_isolation.py tests/composer/test_engine_factory.py tests/composer/test_composer_logging.py --no-cov -q`: 21 passed, five warnings, 30.95 s.
3. After adding explicit dtype rejection and inherited-logging coverage, `python3 -m pytest tests/engine/test_prepared_run.py --no-cov -q`: 11 passed, 13.04 s.

The maintained tests inspect actual native adapter timing boundaries and JAX compilation events, preserve outcomes across seeds `(44,42,43)`, typed/legacy PRNG keys, logging/history variants, and compare genetic best genomes against the ordinary execution path. Ruff and diff whitespace checks passed. Hardware/sharding variants still need the separate accelerator validation gate; no H100 speed claim is made.

The historical audit probe/evidence describe the pre-fix runtime at the audit-baseline commit `430b422e`; maintained regressions now replace its old expected-failure behavior.
