# Phase 3 NSGA-II/NEAT/Hopper composition runner

```bash
/opt/anaconda3/bin/python3 scripts/run_publication_phase3.py \
  --smoke --output /private/tmp/NEW_PHASE3_DIRECTORY --seeds 17
```

`--smoke` forces CPU and uses total P=4, G=2, Hopper horizon=2, one held-out episode, and two conditions: a single population, then two islands of two in a directed ring. The ring performs one migration after generation 1; no final-horizon migration. Expected exit is **0 for actual completed composition**, **2 for recorded failure**. Output includes `report.json`, initial/final genotype arrays, objective vectors/ranks, and a traceback on failure. Fresh output directories are mandatory. No cluster submission occurs.

Full defaults are proposed capability workloads, not approved publication confirmation: total P128, G200, H1000, four islands, migration interval20, explicit total migrants per island/event, 20 held-out episodes, and single/ring/bidirectional/fully-connected adjacency. Non-smoke requires explicit seeds, `--migrants-per-island` and `--heldout-policy all`; neither migration count nor the full held-out campaign budget is silently approved. Only the smoke conditions have local validation unless explicitly stated otherwise. This runner reevaluates the **whole final population** on held-out episodes; this is a different, explicitly counted budget from the earlier optional ten-policy sampling proposal. No publication inference or resource approval is implied.

## Components and script-local glue

Production implementations remain unchanged:

- Native `MOEngine` performs genuine NSGA-II offspring evaluation, merge and survivor truncation.
- `MOPopulation.select` performs rank/crowding tournaments.
- `NeatEmitter` performs existing TensorNEAT-backed structural/weight mutation; an adapter exposes selected `.genes`.
- `NeatInterpreter` evaluates the NEAT genome; existing `RLEvaluator` executes actual verified MuJoCo/MJX Hopper physics. Silent Cartpole fallback is rejected.
- Python orchestration supplies explicit populations and routes migrants. Existing scalar `IslandEngineAdapter` is **not** used. Migrants are selected by existing rank/crowding truncation; receivers merge all incoming populations and then apply existing NSGA-II survival. This refreshes front/crowding metadata.

This is **asexual NSGA-II with NEAT representation/variation**, not a claim of standard speciation-based NEAT or sexual crossover. Mutation innovation slots use generation × number-of-islands + island-index to avoid reuse across island/generation mutation batches. Crossover homology is not exercised or validated.

The objectives both minimize: `[-episodic_return, number_of_active_stored_connections]`. Active means finite first connection attribute, matching the genome mask; it is not reachability-pruned expressed complexity. Every candidate/generation/condition within a seed uses the same training environment episode key. Held-out episodes use fresh common keys. Thus migrated fitness vectors retain the same deterministic evaluation-instance semantics; this is not stochastic robustness evidence.

The same full initial NEAT population is created once per seed and saved/hashed. Single population gets all individuals; island conditions receive contiguous slices. Algorithm keys and partition/mutation batching vary; identical trajectories are not expected. Total population is fixed. Every topology sends the same configured total outgoing individuals per island/event. Ranked migrants are sent once each, with destinations rotated in a deterministic balanced schedule across events; edge routes are recorded. Fully connected adjacency differs from the repository's random elite-pool topology. They are capability variations, not isolated topology benchmarks. When only two islands are used, ring/bidirectional/fully-connected adjacency coincide; the default smoke runs ring only.

## Outputs and limits

Every generation records objective vectors, return/complexity, active node/connection counts, ranks and nondominated counts by island. Final global fronts are **postprocessed**, not a globally shared evolving archive. Training and held-out fronts are separate. Held-out scores recompute the front instead of pretending training ranks survive reevaluation.

Counts include initialization and fixed-scan simulation step slots; slots include masked steps after termination, not live-transition counts. Wall time includes initialization/compilation and is descriptive, not a warmed performance measurement. Publication-local rollout instrumentation records live/valid steps, invalid rewards/actions/observations and monitored q/qd dynamics, reset anomalies and terminations. It excludes terminal padding from descriptors. Intentionally padded internal simulator buffers are not checked; finite objectives alone are not a validity certificate. All optional dependencies must already be available; nothing is installed. No leaderboard, significance or equivalence claims are generated.

## Validated local smoke

Final-source command above with output `/private/tmp/mjax-phase3-e2e-final-20261109` and seed17 completed successfully on CPU: single and two-island ring, two generations each, one ring migration (two transfers), all final policies held-out evaluated, no failures. Parent verified report status, common initial hash and exact runner SHA256. Total elapsed including imports/preparation was about 200 seconds. This supersedes the earlier preflight-only and interrupted obsolete-source attempts.

Reconciliation changes invalidate blanket reuse of the old smoke evidence for new routing/rollout behavior. See the living plan for the revised validation evidence. Script-local glue and optimizer implementations remain separate.
