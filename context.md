# MalthusJAX — publication planning handoff

Updated 9 November 2026 (Europe/Rome). Read this first in the new planning chat.

## Objective and current decision

The repository audit and implementation packages P0–P11 plus timing repairs PT1–PT5 have been completed within their documented scopes. The next task is **publication experiment planning**, not another audit or automatic launch of the old full publication generator.

The user chose **core parity, timing and islands first**, leaving RL/control and XOR as demonstrations outside the first rerun campaign. They prefer practical equivalence for native/QDax quality but **have not agreed a tolerance**. They explicitly requested discussion of alternatives to the proposed ±1.0 objective / ±2 percentage-point coverage package. Do not treat those proposed values as approved. The latest open question is whether equivalence should guarantee similar final objective values or attainment of the same fixed accuracy target under the same evaluation budget. Explain that distinction before asking for numerical bounds.

No publication pilot/confirmation job has been launched or approved by the draft budget alone. The proposed ceilings are 2.5 H100-hours for planning pilots and 10 for confirmation; both are review drafts. No repeated PT5 diagnostic gate is needed after the current-revision success.

## Repositories and branches

- Framework: `/Users/leonardodicaterina/Documents/GitHub/MalthusJAX`.
- Paper: `/Users/leonardodicaterina/Documents/GitHub/MalthusJAX_Paper`. It contains pre-existing user modifications; preserve them. No paper content was changed during P6–P11 or publication design.
- Framework branch: `codex/audit-remediation`, pushed to origin. Do not switch to main or reset the user's checkout.
- Experiment/code revision: `0e1752f24c36b08d149210b6a503f8545effcb99`.
- Publication design checkpoint: `fe9107fa6eedebcf762b2d50e3b31a46027b7a01` (documentation/planning artifacts only). Handoff is a subsequent documentation commit; inspect `git log` for its hash rather than relabelling experiment provenance.
- Cluster's latest successful runs used `/home/20240485/post_submission`, not the earlier `/home/20240485/post_submission_remediation`. User executes supplied commands on the cluster; no remote execution connection has been established in this chat.

## Read in this order

1. [Publication campaign draft](docs/publication_design/README.md): source findings, endpoints, policy alternatives, sample size, timing/island controls, budgets and implementation prerequisites.
2. [Environment notes](docs/handoff/environment.md) and [local snapshot](docs/handoff/local_environment.json).
3. [Cluster experiments](docs/handoff/cluster_experiments.json), especially smoke 1630 and PT5 1649. Their evidence is user-pasted stdout/JSON, not locally fetched raw reports.
4. [Source map](docs/publication_design/source_map.json) and [source census](docs/publication_design/source_census.json).
5. [Blocked policy template](docs/publication_design/quality_policy.draft.json) and [TOST sensitivity](docs/publication_design/tost_sensitivity.csv).
6. [Remediation plan](docs/audit_remediation_plan.md), [ledger](docs/audit_remediation/findings.md), and [P11 evidence/limits](docs/audit_remediation/P11.md). P3/P4/P6/P7/P9/PT5 notes contain metric, inference, island, regression, provenance and timing contracts.

## Key implementation history

| Package | Commit | Scope |
|---|---|---|
| P6 | `a112709b` | Exact island generations, migration boundaries and zero-horizon behavior |
| P7 | `1eab828c` | Unique pairing, explicit transforms, actual fitted regression statistics/covariance |
| P8 | `8e7b4fd5` | Narrow truthful optimization API, rejected unsupported controls, weighted covariance |
| P9 | `db00e104` | Validated outcomes, explicit unavailable values, immutable raw manifests/environment identity |
| P10 | `c4187809` | Paper inventory and deterministic review-only rebuild; historical numbers not validated |
| P11 | `0e1752f2` | Packaged transform, optional import boundaries, wheel/CI/docs assurance |
| Design | `fe9107fa` | Hashed source associations/census, bounded campaign and power sensitivity |

Earlier P1/P3 preserve signed metrics and canonical occupancy-masked QD statistics; P4 uses explicit hypothesis-specific conclusions and policy gating. P5 fixes front-local crowding/crossover contracts. PT1–PT4 retain actual prepared executables, synchronize complete inputs/outputs/effects, propagate preparation failures and compare compatible declared timing phases without dropping unfavorable seeds.

## Validation that must be reported accurately

- Main full CPU run completed with **1,769 passed, two failures, 24 skipped, ten xfailed**. Two fixture failures were corrected; subsequent serial selection covering corrections and QDax oracle/HLO tests passed **24 tests**. Do not describe this as an all-green full rerun.
- Satellite suites: optimize 20, GP 66, NEAT 45 passed; isolated root/satellite wheels and optional-import boundaries passed. GP/NEAT full suites preceded lazy-export changes, which were checked with freshly built wheels afterward.
- Expanded scientific coverage: **48.82%**, measured on stats/benchmarking selection, not full-suite coverage. Unsupported 80% assurance was removed.
- Docker verification is delegated to CI by explicit user choice. Hosted CI outcomes have not been observed in this chat. Do not restart Docker Desktop or claim container validation passed.
- PT5 1516 passed on an earlier revision; current-revision PT5 **1649 passed**, with valid repeated prepared timings across four seed summaries. Full 1649 report/trials have not been copied locally, so do not invent compilation/reference counts from its short stdout.
- Smoke **1630** completed five live datasets/four tables in 657.85 s. Its log showed progress bars, Brax maintenance and legacy-key warnings, with no visible traceback. This is smoke execution evidence, not publication inference.

## Source and design findings to carry forward

- H1 archive: 240 coordinates × 100 seeds × three pipelines = **72,000 execution records**, or 24,000 candidate pairs per baseline contrast. H2: 180 × 100 × ten = **180,000 records**. All 420 coordinate files were fingerprinted. Recorded successes/finite final fitness and no within-coordinate duplicate seeds do not validate historical provenance or statistical claims.
- Neither archive has versioned timing/metric metadata. Quality is eligible only for conditional source/semantic review; legacy timing cannot be repaired by subtracting guessed compilation.
- Candidate archived CSV coefficients round to values in the paper. This establishes association, not that claimed gap transforms, BCa, TOST or corrections were performed. No BCa output/implementation was located in the inspected analysis path.
- Old treatment×log(D) models do not support a universal speedup from `exp(-beta_1)` alone. D=1 is the uncentred reference; report dimension-specific predictions with uncertainty.
- The figure producer explicitly reads the paper's legacy `figures/data/empirical_curves.json`. That/root legacy aggregate lacks per-seed Dataset 5 input for corrected inference. Do not endorse old 1.68x/2.70x claims or copy old tables into the manuscript.
- **Important prerequisite:** empirical Dataset 3 still uses a custom full-interval scan/timer instead of the repaired P6 adapter. Adopt the repaired path and validate exact counts/metadata before island publication runs.
- Current Composer/sweep defaults share one initialization seed across algorithm seeds. The proposed unconditional paired design needs fresh initialization/centroids shared within each pair, explicit seed lists, recorded order and a separate runner. Current seed overrides start at 42; proposed pilot 1000–1029/confirmation 2000+ ranges are not implemented by those flags.
- `--publication` currently launches RL/XOR too and does not encode the proposed design. Do not give it as the new campaign submission command.

## Intended next planning steps

1. Discuss what equivalence should mean and justify tolerance/accuracy target before selecting bounds. Adapter fidelity uses numerical/structural agreement and is distinct from practical native/QDax outcome equivalence.
2. Decide whether coverage is co-primary or descriptive; QD score currently stays secondary with zero-offset higher-is-better occupied-cell sum.
3. Freeze pilot design, meaningful metrics and a variance-based sample-size rule. A proposed separate 30-pair pilot plus confirmation capped at 200 is not a guaranteed-powered study. Pilot outcomes remain visible but do not enter confirmation. No p-value-driven stopping or widening margins to pass.
4. Review scoped runner prerequisites and a concrete bounded GPU submission plan. Freeze manifest/policy/config/source hashes before confirmation.
5. Keep manuscript replacement review-only until corrected evidence exists.

## Collaboration preferences and boundaries

The user is learning agent coding: explain decisions in plain language and make reviewable recommendations, rather than requiring approval for every implementation detail. Routine read-only work and reversible preparation are authorized. Expensive GPU jobs need a concrete configuration/budget and approval. Do not silently select/override physical GPUs: preserve Slurm allocation, distinguish physical GPU 1 from local CUDA 0, and never interfere with GPU 0's other workload. Treat the current chat as handoff; continue the planning discussion in the new chat.
