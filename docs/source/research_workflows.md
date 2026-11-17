# Research workflows

Start with the [quickstart](quickstart.md) to set `workspace` and `wheel`.

## Try EvoSAX and QDax

Install each extra into its own environment. The product wheel declares EvoSAX
`>=0.2.0,<0.3.0` and QDax `>=0.5.0,<0.6.0`.

```bash
for extra in evosax qdax; do
    python3.12 -m venv "$extra"
    "$extra/bin/python" -m pip install "${wheel}[${extra}]"
    "$extra/bin/python" -m pip check
done
```

Save this as `optional_sphere.py`. EvoSAX runs SimpleGA. QDax runs MAP-Elites
with nine archive cells and reports coverage and QD score alongside best fitness.

```python
import math
import sys

from malthusjax.composer import Composer

backend = sys.argv[1]
options = {
    "evosax": {"evosax_strategy": "SimpleGA"},
    "qdax": {"qdax_strategy": "MAPElites", "qdax_num_centroids": 9},
}
result = Composer.create_default().quick_run(
    backend=backend,
    fitness="native_sphere",
    genome_length=4,
    pop_size=16,
    generations=2,
    seeds=(42,),
    output_dir=f"results-{backend}",
    **options[backend],
)
run = result.runs[0]
assert run.status == "success", run.error
metrics = ("best_fitness", "coverage", "qd_score") if backend == "qdax" else ("best_fitness",)
for metric in metrics:
    assert math.isfinite(run.metrics[metric]), run.metrics
    print(metric, run.metrics[metric])
```

```bash
evosax/bin/python -I optional_sphere.py evosax
qdax/bin/python -I optional_sphere.py qdax
```

## Compare a custom objective across search methods

Copy the `examples/corridor_sensors` directory from the repository to a
workspace outside the checkout. Run the following commands from that copy.
The example reuses a custom objective and interpreter for placing two
maintenance sensors along a corridor. The QDax variant adds a descriptor for
the midpoint and separation of the sensors.

```bash
"$workspace/native/bin/python" run.py malthusjax
"$workspace/evosax/bin/python" run.py evosax
"$workspace/qdax/bin/python" run.py qdax
"$workspace/native/bin/mjax" parity sensors.toml
```

`mjax` discovers the objective in `./plugins/`. The TOML compares two
mutation rates. With `shared_initial_population = false` under
`[experiment]`, each backend initializes from the optimizer seed. In this
native comparison, matching seeds produce matching starting populations
across treatments, and different seeds produce different starts. With the
default `true`, supported pipelines receive a fixed generated population
using a separate seed. Inspect the initialization record to see which
policy was applied. Set `capture_population_fingerprint = true` for a
native GA hash of the starting genomes. This copies genomes to the host.

Each CLI invocation writes to one directory. By default this example goes
to `results/sensors`. Choose a fresh path with `--output-dir` for a rerun.
The config snapshot and manifest are under `metadata/`, the seed records
are under `data/`, and `histories_combined.csv` identifies each pipeline
and seed. For example, read the encoded winning genome using the base install.

```bash
"$workspace/native/bin/python" - <<'PY'
import json
from pathlib import Path

path = Path("results/sensors/data/pipeline_low_mutation/seed_11.json")
run = json.loads(path.read_text())
print(run["initialization"])
print(run["solution"])
print(run["metrics"]["best_fitness"])
PY
```

The Python driver decodes the genome into physical positions and checks
`solution.fitness`. With the default `TrackBest.LIGHT`, `solution.selection_scope`
is `final_population_best`. The run's `best_fitness` still reports the best
objective seen across the run. A `TrackBest.FULL` run retains the best-ever genome. For QDax, `export_archive = true` in a pipeline (or
`export_archive=True` in Python) writes an optional NPZ beside that seed's
exports. Its reference is in `artifacts.archive`. CLI references are relative
to the results root. The archive stores occupied cell indices, genotypes,
descriptors, and fitness in QDax's higher-is-better archive convention.

Install the statistics extra in a separate environment to analyze the
two-pipeline results.

```bash
python3.12 -m venv "$workspace/stats"
"$workspace/stats/bin/python" -m pip install "${wheel}[stats]"
"$workspace/stats/bin/mjax" analyze results/sensors
```

The paired analysis writes `analysis/parity_summary.json` and
`analysis/parity_summary.md` inside the results directory.


### Reduce output collection

The default collects history and captures supported solutions. To run a
comparison with terminal metrics and seed JSON, put these options under
`[experiment.shared]` or set them per pipeline.

```toml
collect_history = false
capture_solution = false
write_history_csv = false
```

`collect_history` controls engine collection where supported. Native
Composer adapters reuse the base scan's empty history output. The universal
adapter retains terminal metrics in its history-free loop. An existing
plugin that implements only `run_once(key)` can still be used. The framework
suppresses retained history on request and reports the plugin's collection
as uncontrolled. Use `retain_history = false` to drop rows after collection,
or `write_history_csv = false` to keep rows in JSON without CSV exports.

## Historical benchmark migration note

> **Optimization Direction Convention & Benchmark Re-run Advisory**:
> MalthusJAX enforces a canonical **lower-is-better (minimization)** convention across all engines and evaluators. Evaluators constructed with `maximize=True` negate raw objectives (`-score`), while engines minimize uniformly. Benchmark results involving `BinarySumEvaluator` and `KnapsackEvaluator` (under both `maximize=True` and `maximize=False`) require regeneration following the standardization of evaluation signs.
>
> *Migration notice*: If you were using `maximize=False` on `BinarySumEvaluator` or `KnapsackEvaluator` expecting it to find good solutions, switch to `maximize=True` — the old behavior was accidental due to inverted proxy encodings.
