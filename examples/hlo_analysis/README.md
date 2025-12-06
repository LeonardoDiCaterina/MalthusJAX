HLO Analysis Toolkit
=====================

This small toolkit helps analyze lowered HLO text files produced by JAX
(`jax.jit(...).lower(...).as_text()`), extracting simple metrics useful for
diagnosing compilation structure (nested loops, presence of `threefry`,
`top_k`, `gather`, etc.).

Files
-----

- `hlo_toolkit.py` - core utilities to analyze HLO text and produce per-file
  metrics
- `run_analysis.py` - CLI that searches a directory for HLO files, runs the
  analysis, and writes CSV/JSON outputs

Usage
-----

From the repository root, run (example):

```bash
# analyze HLO files saved in `examples/` and write outputs to `examples/hlo_analysis/results`
python examples/hlo_analysis/run_analysis.py --dir examples --pattern "*hlo*txt" --outdir examples/hlo_analysis/results
```

Outputs
-------

- `hlo_analysis_per_file.json` - per-file analysis (list of dicts)
- `hlo_analysis_summary.csv` - table of per-file metrics
- `hlo_analysis_aggregate.json` - aggregated averages across files

Notes
-----

This toolkit is intentionally simple and heuristic-driven. It gives quick,
actionable signals (e.g., multiple `stablehlo.while` occurrences suggests
nested compiled loops) that helped during the benchmark debugging process
