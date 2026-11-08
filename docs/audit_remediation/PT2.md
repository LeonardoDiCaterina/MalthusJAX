# PT2 — readiness barriers and timer boundaries

Added `core.timing.block_until_ready()`, following registered JAX PyTrees with explicit support for plain dataclasses and SimpleNamespace. Real JAX array waits are batched; device errors propagate, including AttributeError raised inside a wait. The universal compatibility helper delegates to it instead of inspecting arbitrary object dictionaries and swallowing errors.

Native preparation now waits for all initialized inputs before lowering. Base timed execution waits for inputs before a monotonic `perf_counter` start and all returned state/history before its end. Native adapter timing additionally waits for all outputs after engine-specific finalization (including genetic best-genome extraction). Upstream/MO/island adoption follows in PT3.

Regression evidence:

- The first new readiness tests failed collection before the required timing module existed (`/tmp/pt2-red.txt`). The baseline timing audit had independently reproduced skipped custom PyTrees and swallowed nested failures.
- `python3 -m pytest tests/core/test_timing.py tests/engine/test_prepared_run.py tests/engine/test_base_abstract.py tests/test_timing_isolation.py tests/composer/test_adapters_utils.py --no-cov -q`: 31 passed, 12.21 s.
- With two additional regressions exercising the actual universal compatibility wrapper, `python3 -m pytest tests/core/test_timing.py --no-cov -q`: eight passed, 0.20 s.
- Ruff and diff whitespace checks passed.

Tests verify event order rather than CPU speed: pending input before timer/lowering, every state/history leaf before timer end, slots-based registered trees, plain dataclasses, and nested failures. PT1's seeded outcome and donation tests continue to pass. No accelerator speed result or complete PT4 timing schema is claimed.
