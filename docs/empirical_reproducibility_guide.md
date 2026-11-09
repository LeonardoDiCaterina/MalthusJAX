# Empirical execution and evidence review

The generator runs live control, TensorNEAT/QD, island, schedule and native/QDax workloads. Generated measurements and tables are review artifacts. Historical speedups, parity statements and island scaling claims require the source mapping and reruns identified in [the evidence inventory](audit_remediation/P10.md). Smoke workloads verify execution; their sample sizes do not establish publication conclusions. Runtime depends on compilation, integrations and hardware.

Use Python 3.12 with a compatible JAX backend. The supported library range is JAX >=0.11,<0.13; CPU checks used 0.11 and the scoped H100 timing check used the cluster's custom 0.12 build. `requirements/analysis-cpu-candidate.txt` records the tested CPU analysis stack, not an approved GPU publication lock. Metal and alternative GPU combinations have not been validated by this remediation.

Install the extras required by your workload. Full empirical generation uses optional integrations, including simulators; import-only checks do not establish that all environments execute. Record the exact revision and environment before a run. Keep Slurm's assigned `CUDA_VISIBLE_DEVICES`; do not select another user's GPU by free memory.

## Reduced live run

From the repository root, with dependencies installed and a suitable allocated device:

```bash
mkdir -p logs
python examples/generate_empirical_paper_curves.py --smoke \
  --out "results/smoke-$(date +%Y%m%d-%H%M%S)/empirical_curves.json"
```

The output parent must be new or empty. JSON, `tables/` and raw `runs/` stay within that run directory. Existing evidence is refused. The script does not discover or synchronize a sibling manuscript. Signed objectives and instantaneous curves retain their meaning; separately named best-so-far control curves are available. Without `--d5-statistical-policy PATH`, Dataset 5 inference is descriptive only. A policy needs a predeclared estimand, margins, multiplicity family and sample-size rationale; a nonsignificant difference is not evidence of equivalence.

For Slurm, review the script's partition, resources, environment and eight-hour time limit before submission:

```bash
mkdir -p logs
sbatch scripts/submit_empirical_curves.sh --smoke \
  --out "results/smoke-$(date +%Y%m%d-%H%M%S)/empirical_curves.json"
squeue -u "$USER"
tail -f logs/mjx_paper_curves_JOB_ID.out
# Cancel your job if needed:
scancel JOB_ID
```

`--publication` and `--heavy` select larger workloads. They require a reviewed configuration and compute budget; their names do not certify statistical validity. No publication-sized rerun was launched during P6–P11.

## Timing-only validation

This uses a separate, frozen native/QDax workload, preserving every seed and first/repeated execution:

```bash
PT5_REVISION=$(git rev-parse HEAD)
mkdir -p logs
sbatch scripts/submit_timing_validation.sh \
  --expect-revision "$PT5_REVISION" \
  --output "results/pt5-${PT5_REVISION:0:12}-$(date +%Y%m%d-%H%M%S)"
```

[PT5](audit_remediation/PT5.md) records H100 job 1516: 48 timed trials, eight agreeing references, zero execution compilation events. This validates the scoped timing protocol, not all algorithms or publication performance. Comparisons must use compatible declared phases. Preparation failures propagate; unavailable timing is not replaced with total wall time. JAX dispatch is asynchronous: the prepared execution protocol waits for inputs, outputs and effects.

Island execution now performs exactly the requested generations, migrating at completed interval boundaries only. A partial final block has no extra migration; zero generations is a no-op. Historical island timings affected by scheduling need separate validation.

## CPU and packaging checks

```bash
python -m pip install -e '.[dev,stats,evosax,qdax,bbobax]'
python -m pytest --no-cov -q tests/stats tests/benchmarking
python -m pytest --no-cov -q tests
python -m pip install build
python -m build --wheel
python scripts/check_wheel.py dist/*.whl
```

The complete local results, coverage scope and satellite checks are recorded in [P11](audit_remediation/P11.md). GPU checks remain separate. Docker build and CLI checks are delegated to CI at the user's request; local container execution has not passed.

## Publication review

```bash
python scripts/rebuild_publication.py --paper-root /path/to/paper \
  --output /path/to/new-inventory.json
python scripts/rebuild_publication.py --help
```

The review-table mode accepts explicit immutable left/right manifests and a policy, writing to a fresh directory. It rejects unknown provenance and unavailable inference. It does not modify manuscript files. The older `scripts/ingest_cluster_results.py --results-dir PATH --dry-run` is only a read-only H2 CSV/schema checker; it does not regenerate corrected paper evidence. See [P10](audit_remediation/P10.md) for the remaining source mappings, scientific choices and reruns.
