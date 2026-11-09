# Phase 2 runner

`scripts/run_publication_phase2.py` runs quantified RL feasibility, not an
algorithm leaderboard. It uses existing engines/operators and adds orchestration
only. It does not submit jobs, install dependencies, choose a GPU or modify the
manuscript. Full publication workload/metric policies still require review.

From the checkout root, with the intended environment activated:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/run_publication_phase2.py \
  --smoke --output /tmp/NEW_phase2_hopper_smoke --seeds 17
```

This explicitly requests genuine Hopper through MuJoCo/MJX/Brax: the existing
environment factory's Cartpole fallback is rejected. Smoke defaults are population
4, one generation, horizon 2, one held-out evaluation episode and a `(4,)` tanh MLP.
All three conditions run unless `--conditions` selects a subset:
`native_ga`, `evosax_open_es`, `qdax_map_elites`.

The optional `--environment gymnax-cartpole` is a separate lightweight environment
interchangeability example, not a replacement for the Hopper smoke. Its episode
cap is configured in Gymnax environment parameters, not only the evaluator.

Non-smoke defaults are the draft P=128, G=200, H=1000, `(64,64)` tanh MLP,
20 held-out episodes; a non-smoke invocation requires an explicit `--seeds` list.
`--population`, `--generations`, `--horizon` and `--eval-episodes` override dimensions.
These defaults are not a GPU budget authorization or an approved final protocol.

## Recorded evidence

Each seed block generates a fresh uniform population in [-1,1] and passes that
same array to every condition. Its array, hash and centroids are saved. Open_ES
evaluates the supplied candidates during initialization, then initializes its
distribution mean from them; the same candidate population does not imply
identical algorithm states. Every condition verifies its supplied initial-array
hash. Independent algorithm, initialization, centroid and evaluation domains use
separate folded keys.

The native GA's existing initialization evaluates a random population before the
shared-population override. That extra P evaluations is disclosed, not hidden.
Generational evaluations and fixed scanned environment steps are counted
separately. Aggregate live training transitions are not accumulated by the existing engine path and remain explicitly unavailable. Per-evaluation instrumentation and held-out live-step diagnostics are retained where the engine preserves info; fixed scanned slots remain separately counted.

The task evaluator encodes fitness as negative episodic return. The engines
minimize this scalar; held-out episodes are reported in natural return units.
QD converts to its existing higher-is-better archive semantics. Its descriptor is
the first two mean action components, averaged across episode/time dimensions,
normalized/clipped from [-1,1]. Only valid steps before termination enter the descriptor (the valid terminating step is included). Publication-local rollout instrumentation records the mask and anomaly/termination counts. Descriptor substantive meaning still needs scientific review before publication.
The optional discrete Gymnax CartPole task instead uses the frequencies of its
two actions, not a continuous-action descriptor or policy-weight projection.

Final states, selected policy weights, metric histories, failures/tracebacks,
package/source/device identity and a runner source snapshot are retained.
Held-out episode keys are common across contenders within each seed block; policies
are selected from training state before held-out evaluation. One best-return
archive policy is evaluated, not the full archive.

The runner uses existing native preparation/run APIs and the universal adapter's
existing initialization/compiled-loop hooks to retain final states. Public adapter
`run_once` discards those states, so it cannot by itself support held-out policy
evaluation. Hook compatibility is therefore tied to the recorded code revision.

Costs are descriptive single prepared invocations, not repeated performance
comparisons. Publication-local rollout instrumentation records invalid rewards/actions/observations, monitored q/qd dynamics and resets separately from masked fitness. Padded internal simulator buffers are excluded. Finite results alone do not certify absence of anomalies across every training call. Existing
smoke runs establish wiring only, not learning quality or publication readiness.

## Local validation, 9 November 2026

All executions below used `JAX_PLATFORMS=cpu PYTHONPATH=src`, Python 3.12.5,
JAX/jaxlib 0.11.0, EvoSAX 0.2.0, QDax 0.5.0, MuJoCo/MJX 3.10.0, Brax
0.14.2 and Gymnax 0.0.9. Every workload was P=4/G=1/H=2, seed 17, with
one held-out episode. These are wiring tests, not learning results.

| Smoke | Result and retained artifact |
|---|---|
| Initial full Hopper panel | Native GA and Open_ES completed, with held-out returns 1.87644 and 1.95992. The obsolete QD attempt was deliberately interrupted during held-out compilation after training, so the full attempt is **partial**, not an all-green run. Individual completed records: `/private/tmp/mjax-phase2-hopper-smoke-01/seed_17/{native_ga,evosax_open_es}/result.json`. This earlier source hash was `e46d67ce405a94b17b1050473cccbafd4bc414eee41246eda36b5a805103155c`. |
| Corrected Hopper QD-only | Passed training and held-out evaluation with corrected episode/time action-descriptor averaging. Actual wrapper confirmed Hopper pipeline, obs=11/actions=3, no Cartpole fallback. Held-out return 2.03773; three occupied cells of four. `/private/tmp/mjax-phase2-hopper-qd-current-03/summary.json`; runner snapshot retained with source hash `4980e1b4ddb045d82f985bae09940e16ec35412da6e7331ca0dce15d7388de1a`. |
| Gymnax full panel | All three paths passed with finite held-out returns 2.0 and shared initialization hashes, at the same source hash as the corrected Hopper QD smoke. `/private/tmp/mjax-phase2-gymnax-smoke-02/summary.json`. |
| Final Gymnax QD-only | Passed after replacing the discrete task's continuous descriptor with two action frequencies. `/private/tmp/mjax-phase2-gymnax-qd-current-04/summary.json`; exact source snapshot/hash in that directory's manifest. This final change affects the optional discrete environment; Hopper action-descriptor behavior is unchanged. |

All completed Hopper conditions used identical initial candidate hash
`090cb51bb19c5266f5fbdbaab92afd0796d20fd2de4a7534671d8feae62ba373`.
This is complementary validation across revisions, not a claim that the final
source completed an uninterrupted three-condition Hopper run. Ruff and Python
byte compilation also passed. No GPU/cluster jobs were launched.

## Local validation evidence

Genuine CPU Hopper P4/G1/H2 evidence: completed native GA and Open_ES records in `/private/tmp/mjax-phase2-hopper-smoke-01/seed_17`, and completed current QDax record in `/private/tmp/mjax-phase2-hopper-qd-current-03`. All include finite held-out results and the same supplied population hash. The first whole-panel attempt was interrupted during obsolete QD held-out compilation; it is partial, not a successful whole-panel run. Gymnax CartPole also passed an all-three smoke earlier; later descriptor-specific revisions are not automatically covered by that result. These are bounded wiring checks, not task-learning evidence.
