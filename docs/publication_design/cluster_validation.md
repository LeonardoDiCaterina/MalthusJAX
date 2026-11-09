# Bounded cluster validation

These scripts validate executable GPU paths, not publication hypotheses. No job
has been submitted. Proposed request: **one H100, eight CPUs, 32 GB, 30 minutes**.
This is a hard validation ceiling, not a full-workload runtime estimate or approval.
Submit one phase at a time initially if isolating failures is preferable.

Transfer/commit the new runner files first; `git pull` cannot transfer untracked
local files. Activate the intended cluster environment and use the exact deployed
commit as `FULL_COMMIT`. Submit from `/home/20240485/post_submission`.

First inspect a command schedule (no GPU work or submission):

```bash
python scripts/run_publication_validation.py --dry-run --phase all \
  --expect-revision FULL_COMMIT --output results/NEW_VALIDATION_PLAN
```

After explicit approval for the bounded job:

```bash
mkdir -p logs
sbatch scripts/submit_publication_validation.sh --phase all \
  --expect-revision FULL_COMMIT --output results/NEW_GPU_VALIDATION
```

`--phase 1`, `2` or `3` isolates a phase. The launcher uses the submitting
environment, checks Slurm physical allocation 1 and preserves CUDA remapping.
Unexpected allocations exit rather than selecting another device. Workers require
an H100 GPU backend and retain environment metadata, source hashes, commands,
stdout/stderr and exit codes. Failure stops subsequent phases. No implicit retry,
resume, dependency installation or full campaign follows validation.

The GPU validation uses explicit small dimensions **without `--smoke`**, since
local smoke intentionally forces CPU. Phase 3 uses three islands so bidirectional
and directed routing differ. Validation explicitly chooses one outgoing migrant
per island and full-population held-out evaluation at tiny dimensions; these are
validation settings, not approved confirmation policies.

Local CPU checks use each phase script's `--smoke`. Reconciled Phase 3 also supports
`--migrants-per-island`, with each migrant sent once and destinations rotated
across neighbors. Non-smoke requires explicit seeds, migration budget and
`--heldout-policy all`. The sampled-front alternative is not silently selected.

Publication-local rollout instrumentation (`scripts/publication_rl.py`) records
termination/live-step/anomaly counts and supplies QD valid-step masks without
changing optimizer/library source. Live-step masks exclude post-termination
padding; monitored state fields are q/qd plus observations/rewards/actions.
Per-evaluation diagnostics do not imply complete accumulated training diagnostics
where an engine discards auxiliary info.

Full confirmation remains blocked pending task/metric margins, endpoint/family
policy, sample sizing, descriptor/archive review, final held-out budget and a
measured full-shape resource plan. Successful GPU validation is not an equivalence
result, learning result or full campaign authorization. The accepted PT5 gate is
not repeated, and manuscript files remain untouched.

## Exploratory-first run (prepared, not submitted)

`--stage exploratory` loads `configs/publication/exploratory.json`, uses reserved
seeds1000–1002 and marks every result ineligible for confirmation. Phase1 collects
45 training trajectories plus timing repetitions; Phase2 nine Hopper trajectories;
Phase3 twelve composition trajectories. See the living plan for exact dimensions.
Run one phase at a time after tiny GPU validation, with an initial 30-minute hard
ceiling per allocation. This is a cap, not a forecast that each phase will finish.

```bash
python scripts/run_publication_validation.py --dry-run --stage exploratory --phase 1 \
  --expect-revision FULL_COMMIT --output results/NEW_EXPLORATORY_PLAN
# After concrete job approval, from the deployed checkout:
sbatch scripts/submit_publication_validation.sh --stage exploratory --phase 1 \
  --expect-revision FULL_COMMIT --output results/NEW_EXPLORATORY_PHASE1
```

The worker records phase wall time as well as runner-specific costs and retains
partial artifacts on timeout. Shortened horizons/populations expose costs and
output ranges; they do not establish equivalence, useful learning, final power
or reliable linear scaling to the full campaign.
