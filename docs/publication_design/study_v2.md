# Revised study: preparation and release boundaries

The user selected Phase 1 at 50 generations with 100 seed blocks, a task-scoped
ranking question for Phase 2, a fourth cosine GA contender, and front discussion
for Phases 2/3. Phase 1 equivalence bounds explicitly remain pending. Existing
exploratory data are retained as exploratory, not pooled into the larger study.

## Scientific protocol

`configs/publication/study_v2.json` is the versioned proposed protocol.

- Phase 1 keeps the agreed three functions and D10/50; P128/512 remains the
  earlier proposed population panel. There are 100 fresh shared-initialization
  blocks per coordinate, at G50: 6,000 condition trajectories. Report paired
  differences and intervals only. No equivalence statement until meaningful
  margins and the hypothesis family are independently justified. The dispatcher
  requires explicit `--estimation-only` for collection.
- Phase 2 uses the same Hopper definition, fixed MLP(64,64), P32/G20/H200 and
  20 independent blocks (seeds2000–2019), with 20 fresh evaluation episodes per
  selected policy. Four conditions: constant native GA, cosine native GA,
  Open_ES, QDax. Only GA mutation schedule differs: rate .2, initial SD .2,
  constant versus cosine down to .01. No schedule tuning against held-out scores.
  Ranking mode retains the actual historical best GA policy using FULL tracking;
  Open_ES uses its historical best; QD selects best training archive member.
  This changes policy-selection semantics relative to LIGHT feasibility runs, so
  those earlier runs cannot be pooled into the new ranking study.
- Phase 2 primary metric is held-out mean episodic return per training run.
  Six paired contrasts, two-sided tests, Holm family alpha .05; ordinary 95%
  difference intervals are estimation intervals, not simultaneous intervals.
  Condition/seed-factor OLS with seed-clustered covariance is supplementary.
  Incomplete families withhold inference. Higher sample means are descriptive;
  no forced total order when comparisons remain unresolved.
- Phase 2 sample size20 targets useful descriptive precision: ordinary mean CI
  half-width about .47 times across-run SD. It does not guarantee pairwise power.
  The ranking is restricted to this task/wrapper/budget, not a universal library
  leaderboard. Coverage/QD score and training setup/scan costs are secondary.
- Phase 3 retains P32/G20/H200 with 20 blocks per topology, four islands and
  two outgoing migrants per island/event every5 generations (no final event).
  Whole final populations receive20 fresh episodes; this is explicit and costly.
  Report each run's held-out return/active-stored-connection front separately.
  No pooled across-seed front, topology winner or unregistered hypervolume scale.
- Phase 2 return–training-cost plots are configuration trade-off plots. They are
  not evolved multiobjective fronts; cost excludes held-out evaluation. Phase 3
  fronts are recomputed from actual held-out objectives, with dominated policies
  and duplicate objective points retained in source output.

20 blocks and the smaller RL workload are proposed ahead of the larger run,
not demonstrated adequately powered or approved as full standard Hopper training.

## Work units and resource review

`run_publication_study.py` defaults to schedule-only. It expands Phase 1 into
240 units (one coordinate × five seed blocks), Phase 2 into80 single-run units,
and Phase 3 into80 single-seed/topology units. GPU concurrency remains one.
Each proposed Slurm unit is capped at30min, one H100, 8CPU, 32GB. Summed unit
ceilings are **120, 40 and40 H100-hours**, respectively (120+40+40=200
H100-hours at all maxima); these are deliberately conservative per-unit caps,
not runtime estimates or an approved aggregate allocation. Measured exploratory
costs do not establish full-size Phase 1 timing or the cost of20 held-out episodes.
Review representative unit measurements and an aggregate ceiling before launching
the arrays. Never submit all arrays just because a schedule file exists.

Local planning example (replace COMMIT with deployed revision):

```bash
python scripts/run_publication_study.py --phase 2 \
  --expect-revision COMMIT --output results/NEW_STUDY2_SCHEDULE
```

Bounded first cosine unit, only after deployment and resource review:

```bash
mkdir -p logs
sbatch scripts/submit_publication_study.sh --phase 2 --unit 1 --execute \
  --expect-revision COMMIT --output results/NEW_STUDY2_COSINE_UNIT
```

Unit1 is seed2000/cosine, P32/G20/H200, 20 held-out episodes. It is a study run,
not a tuning opportunity; retain it and failures. Additional array recipes should
be issued only after aggregate resource approval. Explicit units avoid repeating
completed configurations after a timeout; automatic favorable-run retries are absent.

## Analysis commands

The analyzer runs on CPU, preserves independent-run units, writes JSON and PNG
outputs to a fresh directory, and does not modify inputs or the manuscript.
Existing pilot reports can already be plotted descriptively:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/analyze_publication_study.py \
  --phase2 results/exploratory-phase2-d2ab991e/phase2/summary.json \
  --phase3 results/exploratory-phase3-d2ab991e/phase3/report.json \
    results/exploratory-phase3-finish-1803-seed1001/report.json \
    results/exploratory-phase3-finish-1803-seed1002/report.json \
  --expected-seeds 1000 1001 1002 --output results/NEW_PILOT_ANALYSIS
```

Pilot Phase2 inference is withheld: cosine is missing, selection mode differs,
and there are three rather than20 held-out episodes. Its descriptive plots remain
useful. For the larger study, supply all80 unit summary files and explicit seeds
2000–2019. No duplicate completed run is silently selected. Supply Phase1 output
directories separately with seeds2000–2099 for estimation-only contrasts.

## Implementation validation

Native constant and cosine genuine Hopper CPU smoke passed at P4/G3/H2 with
shared input hashes and held-out episodes. Configured strength traces were
[.2,.2,.2] and [.2,.105,.01]. Direct mutation-noise checks verify generation
actually changes perturbation scale. Optimizer/library source is unchanged.
Larger GPU data and real report plots require cluster commands: no remote session
exists here, and no new Slurm job has been submitted by this preparation.
