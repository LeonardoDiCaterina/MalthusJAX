# Paper data collection launch

Approved scope: Phase1 G50/100 seed blocks, Phase2 20 blocks × four configurations
and20 held-out episodes, Phase3 20 blocks × four topologies. Phase2 emphasizes
paired effect sizes and return–training-cost trade-offs, not a forced winner.
Phase1 bounds remain pending: collection/analysis is estimation-only. The existing
three-seed exploration is not pooled into this collection.

## Bounded serial submission

`submit_publication_campaign.sh` requests a three-task array with concurrency1:
task0=Phase2, task1=Phase1, task2=Phase3. Each has an eight-hour wall-time ceiling,
one H100, 8CPU and32GB. **Maximum initial allocation is24 H100-hours**, not a
completion forecast. Phase1 and Phase3 costs remain less certain than Phase2.
Submitting these commands accepts those explicit ceilings; no automatic extension
or further allocation follows. Completed units survive budget stops.

```bash
git fetch origin
git merge --ff-only origin/codex/audit-remediation
mkdir -p logs
sbatch scripts/submit_publication_campaign.sh \
  --expect-revision FULL_DEPLOYED_COMMIT --output results/paper-study-v2
```

The jobs preserve Slurm's CUDA remapping and require physical GPU1. The coordinator
does not initialize a GPU outside the allocation. Workers verify H100, capture
environment/source identities and run the unchanged phase scripts. Unit timeout
is30min; scheduling reserves enough allocation time for a whole unit plus a buffer.
When the work budget is exhausted, the phase exits cleanly with ledger status
`budget_exhausted`, **not** `complete`. Other array phases can proceed independently.

## Retention and resume

Phase2 automatically retains the verified unit0/1 results at:

- `results/study-v2-constant-unit0-e83506dd`
- `results/study-v2-cosine-unit1-e83506dd`

Arguments, scientific runner/helper hashes, underlying tracked library changes,
package versions, successful unit manifests and summaries are checked before reuse.
A different source/profile/environment is not silently mixed into the collection.
The existing units retain their original source revision in the ledger.

Each phase writes `results/paper-study-v2/phaseN/campaign.json`. New work uses fresh
attempt directories. Successful units are revalidated before skipping; incomplete
or failed attempts stop collection and require review plus explicit `--retry-units`.
No automatic significance-based stopping, favorable retries, or output overwrites.

After a budget stop, resubmit only the corresponding array index using the same
root and exact revision, e.g. `sbatch --array=1 ...` for Phase1. That creates another
explicit allocation with the same ceiling; it is not authorized automatically by
the first submission. After a unit timeout/failure, inspect its logs first; retries
preserve the original attempt and must identify the unit explicitly.

## Interpretation and analysis

The fixed sample count does not guarantee a significant contrast or equivalence.
Report differences/intervals, failures and missing units. The ranking analyzer uses
all six predeclared paired comparisons and Holm only for a complete eligible family.
Phase2's sample-mean return–cost frontier is descriptive and distinct from Phase3's
evolved return–active-stored-connections fronts. Phase3 fronts stay separate by run.

`analyze_publication_study.py` accepts completed Phase1 directories, Phase2
summary files and Phase3 reports. Use the ledgers' retained result directories,
including old units0/1 exactly once. Supply seeds2000–2099 for Phase1 and2000–2019
for Phases2/3. Incomplete data retain descriptive estimates but withhold the full
Phase2 inferential family. No manuscript replacement or universal algorithm ranking
is produced automatically.
