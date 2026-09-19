# `malthusjax.dash` — Reference

Scope: `malthusjax.dash.catalog`, `malthusjax.dash.plan`, `malthusjax.dash.engine`, `malthusjax.dash.config`, `malthusjax.dash.filters`, `malthusjax.dash.transformers`, `malthusjax.dash.plotting`, `malthusjax.dash.cli`.

---

## Overview

`malthusjax.dash` is the analytical and visualization post-processing engine for MalthusJAX experiment data. It consumes experiment output directories, loads run results into relational dataframes, applies composable data transformers and filters, executes statistical analyses, and renders publication-ready plots.

---

## CLI (`malthusdash`)

The `malthusdash` CLI executes declarative analysis plans from TOML specifications:

```bash
malthusdash run analysis_plan.toml -o ./output_dir
```

### Logging & Diagnostic Flags

`malthusdash` supports global logging flags:

| Flag | Type | Description |
|---|---|---|
| `-v`, `--verbose` | boolean | Enables verbose debug logging (`DEBUG` level). |
| `-q`, `--quiet` | boolean | Suppresses informational progress messages (`WARNING` level). |
| `--log-file PATH` | Path | Routes structured log records to a persistent file. |
| `--log-json` | boolean | Formats console and file logs as newline-delimited JSON objects. |

---

## Structured Logging Channel

- **`malthusjax.dash.cli`**:
  - `INFO`: Plan execution start and successful completion summaries.
  - `ERROR`: Missing dependencies or plan execution failures.
