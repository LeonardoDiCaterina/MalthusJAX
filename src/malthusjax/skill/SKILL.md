---
name: malthusjax
description: Run, compare, and extend MalthusJAX evolutionary experiments with Composer, the mjax CLI, or native JAX engines. Use for typed genomes, reproducible seed studies, or recovering solutions, Pareto sets, and archives.
---

# Use MalthusJAX

Use Composer for Python or TOML experiments. Use a native engine and
`BenchmarkRunner` when supplying a custom evaluator, genome, or emitter.
Build an extension only when a maintained component cannot represent the
problem. Run Composer from a trusted directory. Imports can execute local
Python plugins under `plugins/`.

## Verify a first run

Use Python 3.12 and the installed wheel in a scratch directory. Check
`python -m pip check`, report the installed version and module path, and start
with `JAX_PLATFORMS=cpu`. The base wheel can run this example.

```python
import math
from malthusjax.composer import Composer

run = Composer.create_default().quick_run(
    fitness="native_sphere", genome_length=4, pop_size=16,
    generations=2, seeds=(42,), output_dir="sphere-results",
).runs[0]
assert run.status == "success", run.error
assert math.isfinite(run.metrics["best_fitness"])
print(run.metrics["best_fitness"])
```

The `sphere` and `bbob` presets use EvoSAX. Install the matching extra for an
external backend. Native MAP-Elites uses the `qdax` extra for archive storage.

## Compare experiments

Put common TOML parameters in `[experiment.shared]` and overrides in
`[pipelines.NAME]`. Run `mjax run CONFIG` or `mjax parity CONFIG`. Choose
a fresh destination with `--output-dir` for a repeated run. Inspect
`metadata/run_manifest.json` and the records under `data/` before comparing
successful seeds. By default a comparison can inject a fixed starting
population. Set `shared_initial_population = false` under `[experiment]`
for backend-seeded initialization and verify the recorded policy.
Use the `stats` extra for paired analysis with `mjax analyze RESULTS`.
Compare methods on matched evaluation budgets, including initialization
when it differs by method.

## Recover structured designs

A scalar `best_fitness` and a selected genome can have different scopes.
Use `run.solution["fitness"]` to check the genome's recorded score. Native
GA under `TrackBest.LIGHT` reports the best-ever scalar and exports the best
genome in the final population. Native MAP-Elites exports an archive-best
solution and supports `export_archive=True`. Archive fitness uses the
higher-is-better repertoire convention. Native MO supports
`export_pareto=True` for the final finite rank-zero set, including duplicates.
Check the objective directions before interpreting its vectors.

For example, use the native MO factory with a typed permutation config and
a swap emitter. This toy evaluator scores the placement of the first and
last jobs. The NPZ contains actual integer permutations.

```python
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from flax import struct
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import EngineOutputOptions, build_mo_engine
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.categorical_genome import CategoricalGenomeConfig
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.operators.mutation.categorical import SwapMutation

@struct.dataclass
class JobPlacement(BaseMOEvaluator):
    def evaluate(self, genome):
        return genome.values[jnp.array([0, -1])].astype(jnp.float32)

cfg = CategoricalGenomeConfig(num_categories=4, shape=(4,), is_permutation=True)
engine = build_mo_engine(
    JobPlacement(config=BaseEvaluatorConfig(maximize=False), data=None),
    GeneticMutationEmitter(
        mutation=SwapMutation(mutation_rate=1.0), genome_config=cfg, _batch_size=8,
    ),
    genome_type="categorical", genome_config=cfg, pop_size=8, generations=3,
)
run = BenchmarkRunner(
    engine, output_dir=Path("job-results"),
    engine_outputs=EngineOutputOptions(export_pareto=True),
).run([42]).runs[0]
assert run.status == "success", run.error
with np.load(run.artifacts["pareto"], allow_pickle=False) as result:
    routes = result["genotypes"]
    assert len(routes) > 0
    assert np.all(np.sort(routes, axis=1) == np.arange(4))
    np.testing.assert_array_equal(result["objectives"], routes[:, [0, -1]])
    print(routes, result["objectives"])
```

## Extend and diagnose

- Reuse the genome config across the evaluator, population, and emitter.
  Use permutation-safe operators such as swap mutation and Order Crossover.
  Use `evaluate_qd` for fitness plus descriptors and the scalar `evaluate`
  interface for GA. Copy a population or state before a donated run if you
  plan to reuse it.
- For unsupported or missing exports, inspect `run.output_status` and
  `run.error`. Select a compatible adapter or inspect final state with the
  low-level engine. A legacy `run_once(key)` engine can still return metrics.
- For metrics-only runs, use `collect_history=False` and
  `capture_solution=False`. Keep bulk exports off. Use
  `write_history_csv=False` to retain history without writing history CSVs.
  Report effective collection for custom adapters.
- If configuration validation fails, inspect the named pipeline and its
  operator spec. If the destination exists, use a new directory so different
  seed sets cannot be mixed.
