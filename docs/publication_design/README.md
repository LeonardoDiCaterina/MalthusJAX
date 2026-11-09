# Publication campaign design — review draft

Prepared on 9 November 2026 against framework `0e1752f24c36b08d149210b6a503f8545effcb99`. This is a design and evidence-mapping checkpoint, not a launched experiment or a validated replacement for manuscript results. The paper working tree was read only.

## Recorded decisions and evidence

The user selected **core parity, timing and islands first**, retaining RL and XOR as demonstrations outside this campaign. The user prefers a practical-equivalence question for native/QDax quality, but requested discussion of the proposed tolerances: **no equivalence margin is agreed**. The proposed ±1 objective/±2 percentage-point package is not adopted. `quality_policy.draft.json` deliberately has null margins and cannot pass the inference validator. Minimum sample size and confirmation policy remain unfrozen.

PT5 job **1649** passed at the revision above with `SLURM_JOB_GPUS=1`, local `CUDA_VISIBLE_DEVICES=0`, and valid prepared repeated timings. The user-supplied stdout is preserved in `PT5_job_1649_user_summary.json`; original cluster reports/raw trials have not been copied locally. This scoped gate is accepted without repeating it. Job 1630 is an end-to-end GPU smoke success, not a publication sample or a power pilot at the intended publication budget.

## Source map and historical evidence

`source_map.json` links the empirical artifact groups and the manuscript-wide headline claims to candidate producers and source hashes. `source_census.json` fingerprints **420 coordinate files** across H1/H2 and records their actual record counts and missing contract fields. Reproduce the census from the framework root with `python docs/publication_design/collect_source_census.py`. This association map improves on P10's file inventory; it does not manufacture a validated provenance chain.

- H1: 240 coordinates, 24,000 records per pipeline, three pipelines: **72,000 execution records**, not 24,000 total executions. There are 24,000 candidate pairs for each baseline contrast, subject to verifying identities and exclusions across all sources. No duplicate seed within a coordinate/pipeline and no missing/nonfinite recorded final objectives were found. All records declare success; this is a legacy recorded status, not independent validation.
- H2: 180 coordinates, 18,000 records per pipeline, ten pipelines: **180,000 execution records**, with the same limited finite/status/within-coordinate seed checks. Those checks do not prove archive completeness, correct objectives, or independence.
- Neither archive contains versioned timing records or metric metadata on any execution. Timing values cannot be repaired by subtracting estimated compilation; retained timing claims need new measurements. Quality reanalysis is conditional on recovering evaluator identity, optimum/orientation, source versions and actual pairing. Do not pool archived, generated-fixture and cluster sources by recursively scanning every summary filename.
- Two concrete numerical associations exist: H1 Rastrigin wrapper fitness coefficient `0.000356343...` and p=`0.939324...` round to the paper's `0.00036`/`0.939`; H2 elite-selection Rastrigin runtime coefficient `-0.857275...` rounds to `-0.8573`. The archived CSVs are candidate producer outputs, not proof that claimed gap transforms, BCa intervals, TOST or multiplicity were actually performed. The inspected CSVs lack BCa/TOST output, and current analysis code contains no BCa implementation.
- The paper plotting script explicitly reads `figures/data/empirical_curves.json`; that and root `results/empirical_curves.json` are fingerprinted in the census. Both carry the same 7 November metadata and lack Dataset 5 per-seed records. Compare the hashes rather than assuming byte equality. Legacy aggregate curves do not supply paired variance or corrected QD semantics.

No current table/figure is marked publication-valid simply because matching numbers were located. The existing P10 inventory remains the comprehensive line/hash index; the present map covers the substantive empirical groups and shared headline consumers.

## What equivalence should mean

Three distinct questions must not share one margin:

1. **Adapter fidelity:** does the adapter execute the intended upstream algorithm? Compare actual states, archive contents and histories under the same upstream implementation, keys and inputs. Provisional engineering tolerances (`atol=1e-6`, `rtol=1e-5` for float32 finite numeric leaves) need leaf-specific validation; integer/topology identities and occupancy masks must match exactly. These are numerical regression criteria, not a statistical equivalence margin or a claim across hardware. A failure is investigated, not fixed by widening tolerance after the fact.
2. **Native/QDax practical outcome equivalence:** does a mean quality difference small enough to matter lie within predeclared bounds? Native and QDax can differ in insertion, selection and evaluation work, so this is a separate empirical question. Rastrigin optimum zero makes percentage differences relative to attained best fitness unstable. A defensible absolute tolerance should follow a chosen accuracy target/use case. A fixed, externally justified reference scale could support relative gaps, but the pilot/sample mean cannot set that reference after observation.
3. **Task success:** do both methods attain a predeclared objective threshold within an equal evaluation budget? This is often easier to interpret than an arbitrary mean margin, but changes the endpoint to success probability/evaluations-to-target and requires a different analysis/implementation. It is not supported by silently passing a transformed vector to the current archive-policy schema.

Candidate raw-unit tolerances for discussion are 0.1, 0.5 or 1.0 objective units, and 0.5, 1 or 2 **percentage points** of coverage (not relative percentages). These values have no universal substantive justification on their own. With 100 centroids, one occupied cell is one percentage point in a single run; a mean can still take fractional-cell increments. Best objective can be similar while coverage differs, and vice versa. QD score is a secondary diagnostic using the versioned occupied-cell sum with zero offset, explicitly retaining its negative/higher-is-better meaning. Its coverage/quality coupling prevents using it as an interchangeable fitness endpoint or choosing an offset to improve a narrative.

Recommendation: retain adapter fidelity as a strict engineering check. For native/QDax, define the objective accuracy target first and decide whether coverage is co-primary or descriptive. Keep equivalence unavailable until those choices justify tolerances. A non-significant difference never substitutes for equivalence, and no noninferiority claim is substituted without changing the registered question.

## Proposed native/QDax policy, conditional on agreement

- Workload: 10D Rastrigin, `P=100`, `G=500`, 100 centroids, float32, bounds [-5.12,5.12]. Confirm exact evaluator instance/optimum, descriptor map, centroid generation and mutation/bounds handling before freezing. Retain actual evaluation counts including initialization and archive work.
- Two proposed co-primary endpoints: signed final best objective in original minimization units and coverage on a 0–100 scale. Observation is final instantaneous endpoint; best-so-far curves are separately labelled. Native minus QDax paired mean difference is the estimand.
- A seed pair shares fresh initial population/centroids within the pair. Initialization and algorithm/environment keys are recorded independently and varied across pairs. Current Composer/sweep defaults share one `pop_seed` across all seeds; that can support a conditional-on-initialization estimand, not the intended unconditional sampling design. The new runner must implement and record per-pair initial populations; do not pretend this already happens.
- Paired mean TOST, family alpha .05, Holm across the two declared equivalence hypotheses. The primary quality claim requires **both** adjusted conclusions to establish equivalence. If either is unavailable or inconclusive, say so. Difference tests and QD-score comparisons are secondary/exploratory, not extra routes to the primary claim.
- Report both paired means, paired SD, raw and adjusted TOST p-values and intervals. Show ordinary 95% paired mean intervals plus conservative family intervals at 97.5% confidence for each of two endpoints; they are distinct from the unadjusted 90% TOST interval. Keep raw failures and missing pairs visible. A family claim is withheld if execution attrition prevents analysing the planned estimand; paired-success-only summaries are explicitly descriptive sensitivity analyses.
- Repetitions and generations are not extra independent observations. Do not count two algorithms' outcomes as two independent seeds. No optional stopping based on p-values or choice of whichever final seed looks best.
- Mean-TOST assumptions must be evaluated under a predeclared diagnostic/sensitivity plan; normality non-rejection is not certification. Bounded/discrete coverage, ceiling effects and degenerate variance may make inference unavailable. A bootstrap diagnostic cannot silently replace the primary test after seeing an unwanted result.

## Sample size and precision

A separate **30-pair planning pilot** at the actual workload supplies paired variability and elapsed-time measurements. It does not supply a favorable equivalence margin. Pilot seeds 1000–1029 are reserved for planning; all outcomes remain visible, but none enters the confirmation test. Confirmation seeds start at 2000. The runner must implement these seed ranges; current empirical CLI seed overrides always start at 42 and cannot implement this design unchanged.

Before confirmation, freeze one sample size between 30 and a proposed cap of 200 pairs. Plan conservatively at per-endpoint alpha .025 and target >=90% power per endpoint, giving a union-bound >=80% probability of establishing both under the modeled alternative. Use true mean differences at zero and half the margin as sensitivity scenarios; the half-margin case is the proposed sizing scenario. Use an upper variance estimate rather than the most favorable point estimate, with the normal-theory limitation explicit. If no n<=200 supports the target, revisit resources or report the precision-limited question; **do not widen margins or repeatedly add seeds until significant**.

`tost_sensitivity.csv` is a planning calculation integrating the paired t-TOST rejection region over normal-sample chi-square variance. It is not an empirical power guarantee. Reproduce with `python docs/publication_design/compute_tost_sensitivity.py`. For paired SD twice the margin, n=50 gives about 41% power at true difference half the margin; n=200 about 94%. For paired SD four times the margin, n=200 gives only about 42%. Thus neither 20, 50 nor 200 is a universal adequate sample count. The two-seed smoke outcomes at a different horizon are not used to size confirmation.

## Timing study, without repeating the protocol gate

Apply the accepted prepared-execution protocol to the actual publication configurations. This collects new performance evidence; it is not a repeat of job 1649's gate workload. Record preparation, first prepared execution, repeated prepared execution and end-to-end separately. Use the same history/logging, evaluator, dtype and metric work on both sides. Native/QDax runtime is an implementation-level comparison with archive work reported, not pure framework overhead unless workloads are demonstrably identical.

Proposed QD timing panel: five fixed timing seeds (3000–3004), seven separately launched/process-isolated sessions on physical GPU 1, one first and five repeated prepared calls for each backend/seed per session. This is 420 scan invocations, not 420 independent seeds. Counterbalance order by session and seed with a recorded permutation; never delete first or slower trials. Predeclare persistent cache policy and record clocks/power/device UUID/other-process activity. Separate compilation comparisons require controlled cold-process/cache runs; an ordinary preparation duration is not necessarily cold compilation. Device failures/compilation contamination retain a failed session and stop publication timing acceptance rather than being dropped.

Primary timing display: per-session median of within-seed QDax/native repeated-time ratios, then median across sessions; use a session-level resampling interval with its limited seven-session scope, and publish every session ratio. Do not treat this as a cross-hardware confidence interval or require a favorable speedup. The existing harness summary uses a ratio of backend medians; label it separately if retained, since those estimators differ. Adapter/raw timings use the same blocked scheme on selected fixed coordinates after parity agreement.

## Core adapter panel and historical regression review

Proposed adapter panel: EvoSAX SimpleGA adapter versus an independently invoked raw upstream reference on six named BBOB functions, D={10,50,100}, P={128,512}, G=200: 36 coordinates, ten independent paired seeds, two implementations = **720 trajectories**. This validates the specific adapter; it does not re-establish the old custom `malthusjax_wrapper`/batched-mimic algorithm claim or substitute an ES strategy for a GA. Fix exact upstream versions, strategy parameters, bounds, key consumption, stopping, evaluator offsets and summary work in manifests.

Historical H1 quality may be reanalysed separately only after identity/orientation/optimum recovery. Use one-to-one identity including coordinate and seed, preserving actual replicated coordinates. Do not log signed fitness blindly or add a sample-derived shift. Predeclare raw objective or `log(f-f*+epsilon)` with true task optimum, scale and epsilon. A paired within-coordinate contrast model avoids treating duplicate backend rows as independent. For a grid-wide regression, coordinate dependence and repeated seed IDs need an explicit clustering/resampling design; HC3 alone does not address these dependencies. Use crossed seed/coordinate resampling or justified cluster analysis, with finite cluster counts reported.

The old interaction model has a treatment×log(D) term: runtime ratio varies with D. `exp(-beta_1)` alone is a ratio at log(D)=0 (D=1), outside the sampled range, not a universal speedup. New fits should centre covariates at a declared reference (e.g. D=50) and report `exp(-(beta_T+beta_TD*log(D/D_ref)))` with intervals at actual coordinates. Do not claim BCa bootstrap unless the implemented resampling unit, algorithm and generated intervals are saved and verified.

H2 operator ablation and H3 precision are **not** GPU rerun priorities in this campaign. Preserve and analyse their source chain on CPU first. Changing tournament/elite pressure, mutation distributions or elitism confounds architectural speed attribution; same distributions/pressure must be demonstrated before causal overhead claims. Headline 2.70x claims remain withheld until supported by a dedicated fair comparison. H3 nonfinite precision failures remain failures, not filtered successes.

## Island experiment and confounding controls

Use the P6 `IslandEngineAdapter` exact-horizon prepared path. The current empirical Dataset 3 function still has a separate custom full-interval scan and timer; it is not publication-ready merely because P6 passed. Do not launch its existing island table generator unchanged.

Two panels distinguish resources:

- Strong scaling: total population 1,024, M={1,4,16,64}, per-island population {1024,256,64,16}.
- Weak scaling: per-island population 128, M={1,4,16,64}, total population {128,512,2048,8192}.

D=20 Rastrigin, G=500, migration interval 25. For each M>1 include migration on and off under the same partition/local engine/elitism/initial arrays; M=1 has one baseline per panel. This gives **14 conditions**, not 16 after removing redundant M=1 migration comparisons. Use ten paired seed blocks per condition; record per-island initialization, migrant count and work. Migration off means zero migrants under the same scheduled path, not an unrelated monolithic algorithm. Choose a common fixed migrant count of two per island so migration rate changes with island size are visible; changing fractions is a separate design factor.

Strong/weak scaling timing is descriptive resource scaling, not proof of asymptotic O(1) compilation. Compile-time variation needs cold-process replication; five isolated compile trials per distinct signature are proposed. M changes total selection/elitism and topology, so monolithic-versus-island quality cannot by itself prove a migration benefit. On/off within a partition is the relevant migration contrast. Plot actual pre/post-migration measurements if claiming migration-caused breakthroughs; do not align smoothed best-so-far jumps and call that causal evidence. Initial and generational evaluations are counted separately and fairly. No multi-GPU or multi-node scalability claim follows from this single-GPU panel.

## Proposed hard resource ceilings (not runtime forecasts)

One H100 allocation at a time; no concurrent CPU/GPU jobs using the same device for timing. GPU 0's vLLM workload can still affect host/thermal conditions: record it and limit claims to the observed shared-node setting. Scheduler allocation and local CUDA numbering remain distinct.

| Stage | Planned workload | H100 wall-time ceiling | Release condition |
|---|---|---:|---|
| Source review and analysis planning | Archived source checks and simulations | CPU only | No GPU submission |
| QD planning pilot | 30 pairs, publication P/G/C | 30 min | Margins/endpoints/design frozen; runner ready |
| Adapter pilot | 9 fixed coordinates × 3 pairs × 2 | 1 h | Raw-reference and adapter workload agreement |
| Island pilot | 14 conditions × 3 seed blocks | 1 h | P6 path adopted; exact counts and readiness checked |
| QD confirmation + timing panel | Fixed n<=200 pairs + 420 timing scan invocations | 4 h | Pilot variance/runtime reviewed; policy/n preregistered |
| Adapter confirmation + matched timing | 36 coordinates × 10 pairs × 2 | 4 h | Pilot passes; block order/costs frozen |
| Island confirmation/cold-compile panel | 14 conditions × 10 seeds + 5 compile trials/signature | 2 h | Pilot supports ceiling and input/output equality |

Maximum initial planning-pilot envelope is **2.5 H100-hours**; confirmation envelope **10 H100-hours**; total **12.5**, not an authorization or a prediction. The much larger historical 72k/180k execution campaigns are not automatically relaunched. Calibrate runtime per shape, include compilation/reference checks/I/O, and keep >=2x headroom from observed pilots. A Slurm timeout creates an incomplete dataset; preserve it. Revisions/configurations may not change mid-campaign. Completed valid observations stay retained and no favorable-run-based retry policy is allowed.

## Implementation and approval boundary

Before any pilot: implement scoped dataset selection, explicit seed lists, per-pair initial-population/centroid snapshots, run-order scheduling, exact-path island collection and per-seed compile/timing/evaluation metadata. PT5's known-good machinery should be reused. Keep the frozen PT5 configuration unchanged. Add raw-reference and deterministic schedule checks appropriate to these additions. Existing `--publication` launches unrelated RL/XOR and does not encode this plan; do not use it as the submission recipe.

Before confirmation: agree the substantive equivalence claim/margins, document the pilot-based fixed n and assumptions, freeze/hash the exact manifests/policies/configs, verify environment/import paths and known dependency/CI limitations, then approve bounded jobs. Failed fidelity checks, invalid metric contracts, unknown optimum, missing source identity, contamination or inadequate compute stop the corresponding claim. Inconclusive equivalence is publishable as inconclusive; no requirement asks results to favour MalthusJAX.

## Method references

- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and [asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html): compilation, transfer and synchronized execution boundaries.
- [Lakens, equivalence-testing primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/): equivalence bounds define practically meaningful effects and require explicit justification; non-rejection of a difference is insufficient.
- [statsmodels regression implementation](https://www.statsmodels.org/stable/_modules/statsmodels/regression/linear_model.html): covariance choices including clustered groups; the appropriate grouping remains a design decision, not a software default.

## Local design validation

All 420 raw-coordinate hashes and mapped artifact/source hashes were rechecked. Both planning helpers pass Ruff F checks. The 25 power scenarios are bounded probabilities; an independent 150,000-sample normal Monte Carlo check at n=50, paired SD/margin=2 and true difference half the margin gave 0.40969 versus analytic 0.40967. This validates that planning calculation under its assumptions, not the assumptions for actual optimisation outcomes. The current policy template was passed to `validate_policy` and correctly rejected because its margins are unset. No production code or manuscript content was changed by this design checkpoint.
