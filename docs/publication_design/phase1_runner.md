# Phase 1 runner

`run_publication_phase1.py` collects quality outcomes and existing engine timing records. It does not submit jobs or infer equivalence. Optional `--timing` collects retained-executable first/warm/repeated measurements; independent process sessions and hardware controls remain a cluster-orchestration responsibility.

From the checkout root:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src MPLCONFIGDIR=/private/tmp/mjax-phase1-mpl \
  /opt/anaconda3/bin/python3 scripts/run_publication_phase1.py \
  --smoke --seeds 17 18 --output /private/tmp/NEW_PHASE1_SMOKE
```

`--smoke` forces CPU, Sphere D=4, P=16, G=2, nine uniform archive sites, and at most three seed blocks. All five contender paths are exercised unless `--library evosax` or `--library qdax` is selected. Existing output directories are rejected. No packages are installed. Ordinary mode exposes `--functions`, `--dims`, `--populations`, `--generations` and explicit `--seeds`; its default coordinate grid is proposed, not an approved campaign.

Each seed block generates one fresh population via fold-in 101 and one centroid set via fold-in 102. Population/centroid `.npy` files and SHA256 hashes are retained. Every condition receives the same initial array. Uniform random sites are explicitly used, not CVT. Contender order rotates by seed, is recorded, and is not a randomized timing-panel schedule.

## Contenders and boundaries

- Adapted EvoSAX SimpleGA: crossover .3, elite ratio .16666666, default constant mutation SD 1.
- Native standard/batched H1 wrappers: mimic elite selection, imported EvoSAX crossover .3 and mutation SD 1, elitism 0. Populations with a mismatch between H1 `max(2,int(P/6))` and upstream `max(1,int(.16666666*P))` are refused. P=16 smoke and proposed P=128/512 match. This explains why P=12 was discarded after initial wiring validation.
- QDax adapter: upstream MixingEmitter; native MAP-Elites: existing replica emitter. Both receive the same Gaussian mutation SD .1, per-gene uniform crossover, variation fraction .5, centroids and initial population. This matches the intended selected operators; it is not evidence that every archive/tie/engine detail is already scientifically validated.
- QD descriptors use existing `GenotypeBoxDescriptorFn`: first two genes, normalized and clipped to [0,1] using bounds [-5,5]. These are genotype descriptors, not task behaviors.
- Fitness uses seeded EvoSAX BBOB instances (`seed=42`), not necessarily zero-optimum unshifted toy functions. Negative raw objectives can be legitimate shifted BBOB outputs; no optimum claim is made.

The manifest records script hash, revision, package versions, devices and settings. Raw per-condition records preserve metrics, history, errors and engine timing contracts. Search-budget accounting separates initial P evaluations, reported generational evaluations and scheduled P×(G+1) total; compilation/preparation may perform additional work outside the search budget, so these are not device-instrumented call counts. Supplied array hashes prove the orchestrator shared inputs, not universal internal-state equality.

Failure conditions remain visible and return nonzero exit status. `summary.json` deliberately sets `publication_ready=false`: margins/sample-size policy, exact settings/metric correspondence and independent reference review are still required. Collection wall time includes construction/preparation; existing first-prepared timing records are not relabeled steady-state measurements.

## Local validation on 9 November 2026

CPU smoke command above with `--seeds 17 18 --output /private/tmp/mjax-phase1-smoke-final` completed all **10/10** condition runs (five contenders per block), with finite best fitness, generation 2 and reported generational evaluations 32. Saved input hashes were identical across conditions within each block and differed between the two seed blocks. Both resolved elite counts were 2. Search-accounting total was 48 including 16 initial evaluations. These checks do not assert empirical quality equivalence.

Environment: Python 3.12.5, JAX/jaxlib 0.11.0, EvoSAX 0.2.0, QDax 0.5.0 on local CPU. Ruff F checks, CLI `--help` and refusal of an existing output directory passed. Fontconfig cache warnings appeared during import; execution completed. An earlier P=12 smoke was wiring-only and discarded because upstream/native elite rounding differed; it is not the final smoke configuration.

## Optional dedicated speed collection

Add `--timing --repetitions 5` to collect speed evidence. Smoke caps repetitions at two. The runner builds the actual adapter, observes its existing `TimingCapture.initialized` hook and stops before ordinary execution. It compiles the captured universal loop or native `prepare_run` exactly once, retains that AOT executable, then directly invokes it with fresh copies of the exact initialized inputs. Input copies are synchronized outside the measured interval to accommodate native state donation.

Every condition retains one first prepared call, one warm-up call and all repeated calls, synchronizing complete outputs and effects. Initialization/preparation times, input signatures, reset policy, scan evaluation denominator, JAX compile/tracing log observations and exact repeated-output agreement are recorded. Execution observations invalidate timing on logged compilation or output disagreement. Logging is scoped evidence; it is not a universal compilation detector. AOT invocation cannot silently retrace the retained callable.

`evaluations_per_scan / seconds` gives scan throughput, excluding initialization. Input reset and host validation are excluded from scan timing and must not be presented as end-to-end throughput. There is no automatic outlier deletion, seed averaging or speedup assertion. CPU smoke timings are engineering checks, not publication GPU performance results. Use isolated cluster sessions and matched output work for inferential timing claims.

Dedicated CPU timing smoke (`--smoke --timing --seeds 17 --output /private/tmp/mjax-phase1-timing-final`) passed all five contenders: **20 retained-executable calls** total (first + warm-up + two repeats each), exact repeated same-state output agreement, no observed execution compilation/tracing log events. The initial timing smoke exposed typed-key NumPy conversion in the validator; this was corrected with `jax.random.key_data` and all paths rerun successfully.

Setup-work caveat: existing `GeneticEngineAdapter.run_once` first initializes/evaluates a random population, then reevaluates the explicitly supplied shared population. The runner records this source-derived extra **P discarded setup evaluations** for standard/batched GA in addition to the logical search budget. This affects total collection work and end-to-end cost, not the retained scan denominator. Optional timing capture independently initializes its state and performs extra scans; its work must not be mistaken for extra independent quality seeds. No production implementation was altered to remove this overhead.
