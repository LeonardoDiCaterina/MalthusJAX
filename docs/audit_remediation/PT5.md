# PT5 — Timing-only cluster validation

Status: implemented and validated on CPU and the H100 native/QDax workload. **Timing gate B passes within this workload, based on user-provided job 1516 output and report inspection.** No RL-heavy or publication rerun was launched; historical job 1303 remains unchanged.

## Workload and output contract

`configs/timing/pt5.toml` freezes Rastrigin D=10, population=32, generations=15, 20 centroids, seeds 42–45, no additional preconditioning, and five reset-state repeats after the separately retained first execution. `scripts/validate_timing.py` builds the production native MAP-Elites and upstream QDax adapters through the same factory used by Composer. Both receive the same fixed initial population. Backend order alternates according to seed/trial indices and is serialized before measurement; it never depends on observed times. Each trial starts from the same seed and a fresh initialized state while retaining executable caches.

There are **48 timed trials** (four seeds × two backends × six executions), followed by **eight references**. References run only after all timed trials. Native reference execution invokes the ordinary JIT path with `time_it=False` and no retained prepared executable; upstream reference execution uses `run_once(compile=False)`. The latter retains its JIT callable, rather than an AOT compiled object. Reference adapter phase durations are diagnostic only and never enter performance summaries. Each timed trial's complete reported summary/history must agree with its same-seed reference (rtol=atol=1e-6, equal NaNs allowed for optional statistics). This is a within-backend contract; cross-backend trajectory equality and QD-score comparability are not assumed.

Phase observers use the PT4 capture boundaries and retain initialization/preparation, execution, and host-postprocessing events with backend, seed, trial ID, and global run order. Their execution observation window conservatively begins at the preparation-complete marker and ends at the post-execution marker. JAX tracing/compilation logs are phase-tagged. Any such signal inside a measurement's execution window makes its timing unavailable and fails validation. References are labeled separately and cannot contaminate this decision. The gate also requires at least one captured compilation signal outside measured execution, so a silent log observer cannot pass. Phase events, valid producer records, identical per-seed executable/input signatures across trials, and seeded output agreement are all checked. Exceptions produce a failed report and a nonzero exit; no preparation or synchronization error is silently accepted.

Output must be an absent directory. Existing directories are refused before opening logs. Artifacts include:

- `manifest.json`: frozen configuration/schedule, exact source revision/import path, entrypoint/config hashes, JAX/jaxlib/QDax and installed CUDA-related package versions, scheduler/device environment, and NVIDIA driver/utilization snapshot.
- `raw_trials.jsonl`: append-only observations as collected, including every first execution and repeat. This survives a later failed check; collection-time status precedes reference validation.
- `compilation.jsonl`: phase-tagged JAX logs, including preparation/reference compilation.
- `report.json`: all raw observations, phase events, reference results and agreement flags, validation problems, compilation diagnostics, descriptive repeated-phase medians/counts, and the observed QDax/native time ratio even when unfavorable.
- One directory per backend with normal summary/history exports and `timing_trials.csv`; final timing validity includes post-collection reference/phase checks.

The repeated-phase summary takes each seed's median across its five repeats, then the median across seeds. Repetitions do not increase the independent seed count. No speed threshold is a pass condition and no inferential speedup/equivalence claim is made. This small workload cannot exclude device contention or establish general steady-state throughput. EvoSAX/TensorNEAT, MO/island GPU paths and scaling sweeps remain outside this job.

## Slurm submission prepared for approval

The new wrapper requests **one H100, eight CPUs, 32 GiB RAM, and a 30-minute wall-time cap**. This is a resource cap, not a demonstrated runtime estimate. It uses the submitting shell's activated Python environment, prepends this checkout's `src`, forwards `"$@"` without splitting quoted paths, and preserves `CUDA_VISIBLE_DEVICES` exactly. It never searches other GPUs for spare memory. The script checks that JAX sees one H100 GPU, the expected full revision, and this checkout's imported package. Revision mismatch or CPU fallback fails before adapter construction. Capture the expected revision at submission time; changing the checkout while the job is queued then causes an explicit failure.

The existing empirical Slurm wrapper also forwards all arguments unchanged, preserves scheduler GPU assignment, and logs source/Python identity. Its generator's required output option remains `--out`; PT5 uses `--output`.

After GPU-run authorization, execute on the cluster from the remediation checkout:

```bash
git fetch origin
git switch codex/audit-remediation
git pull --ff-only origin codex/audit-remediation
conda activate post_sub
mkdir -p logs
PT5_REVISION=$(git rev-parse HEAD)
sbatch scripts/submit_timing_validation.sh \
  --expect-revision "$PT5_REVISION" \
  --output "results/pt5-${PT5_REVISION:0:12}-$(date +%Y%m%d-%H%M%S)"
```

Record the returned job ID. Then inspect `squeue -u "$USER"`, `logs/mjx_pt5_timing_JOBID.out`, and `.err`. The actual diagnostic gate is `report.json`'s `status`, compilation-event list, retained trial counts, and reference agreements, not Slurm completion alone. Keep a failed run intact; any follow-up uses a fresh directory and records its configuration before observing durations. A successful CPU run does not close GPU gate B.

## Local verification

The local CPU evidence JSON retains every timed trial's numeric durations/phases, validation status, signature hash, the fixed schedule, reference agreement results, package/source identity, script hash, and selected summary. The source revision identifies the PT4 parent plus a dirty working tree; this is explicitly candidate-code evidence, not an assertion that the parent commit contains PT5. The entrypoint hash identifies the exact tested script.

Commands used (output directory absent before each run):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/bin/python3 scripts/validate_timing.py \
  --output /tmp/mjax-pt5-cpu-verified \
  --expect-revision 7e79d98b926455571375acbc8a263897287aa5e9 \
  --expect-backend cpu --expect-device CPU

/opt/anaconda3/bin/python3 -m pytest --no-cov -q \
  tests/test_pt5_validation.py tests/core/test_timing_records.py \
  tests/composer/test_timing_contract.py

bash -n scripts/submit_timing_validation.sh scripts/submit_empirical_curves.sh
/opt/anaconda3/bin/python3 -m ruff check scripts/validate_timing.py tests/test_pt5_validation.py
git diff --check
```

The regressions cover the frozen alternating schedule, seed 0 and retained trial counts, compilation-detection positive and negative controls, reference disagreement, failed-record exports, revision mismatch before construction, fresh-directory refusal, and quoted argument/GPU assignment preservation for both Slurm wrappers. Existing prepared-executable and timing-record contracts are included. Detailed final counts and CPU gate observations are recorded in `PT5_cpu_evidence.json`. No full repository coverage threshold or GPU pass is claimed.

Final local result: **47 tests passed**, two bounds-default warnings, 28.66 s. The actual production-adapter CPU diagnostic **passed**, retaining **48 timed trials, eight references, and 168 phase markers**, with all eight reference groups agreeing across all six trials and zero measured-execution compilation events. Ruff, shell syntax checks, and diff whitespace checks passed.

## H100 validation — job 1516

After explicit user authorization, the user submitted job **1516** from the remediation checkout. The job reported `passed` at revision `aa6274dbf7176e5b87a34b1a2b3acb19780ac710`, using the intended imported source path, JAX/jaxlib 0.12.0.1, Python 3.12.16, NumPy 2.5.4, and one NVIDIA H100 NVL. Its output directory is `results/pt5-aa6274dbf717-20261109-063346`. User inspection of `report.json` confirmed **48 timed trials, eight references, zero execution compilation events, and all reference agreements true**. This satisfies timing gate B for the declared native/QDax workload. The inspected output is transcribed in `PT5_cluster_1516.json`; complete raw artifacts remain on the cluster and were not independently fetched to this checkout.

The repeated-phase median across four within-seed medians was **0.0015826839953660965 s native** and **0.001887401333078742 s QDax**, each covering the complete 15-generation execution, not a single generation. The observed QDax/native ratio is approximately **1.193**. This is descriptive diagnostic evidence, without a statistical speedup or publication throughput claim.

The job-local NVIDIA snapshot listed GPU UUID `GPU-32603e59-831b-b60c-6cde-1c48e387f37f`, 561 MiB used, and 0% utilization at sampling. The earlier interactive vLLM snapshot therefore does not establish contention during this job. Conversely, a startup snapshot cannot exclude transient contention throughout measurement. The environment records `dirty=true` with the SHA-256 of an empty tracked diff. This is consistent with untracked files; no complete untracked-file inventory was supplied, so the record is preserved rather than relabeled clean. Full artifacts should be archived before using results in the paper. Scientific P3/P4 work remains open.

## Subsequent current-revision validation

User-supplied stdout reports job **1649** passed on `0e1752f24c36b08d149210b6a503f8545effcb99`, with physical Slurm GPU ID 1 mapped to local CUDA 0 and valid `prepared_execution_repeat` timings across four seeds. Median native/ QDax scan times were 0.0015328715 / 0.0018553366 seconds. The stdout summary is preserved in `docs/publication_design/PT5_job_1649_user_summary.json`; original cluster report/raw trials are not locally archived and no extra compilation/reference counts are inferred from this excerpt. The protocol gate need not be repeated to design the publication campaign; new workload-specific performance evidence remains necessary.
