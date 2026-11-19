# Quick Start

MalthusJAX requires Python 3.12 or newer. The wheel declares JAX
`>=0.11.0,<0.13`. These shell commands use a CPU and a fresh virtual environment.

## Build and install the wheel

From the repository root, build the wheel and record its absolute path. Then
switch to a temporary directory for installation and your first run.

```bash
workspace="$(mktemp -d)"
python3.12 -m pip wheel --no-deps --wheel-dir "$workspace" .
wheel="$(find "$workspace" -name 'malthusjax-*.whl' -print)"
cd "$workspace"

python3.12 -m venv native
unset PYTHONPATH PYTHONHOME
native/bin/python -m pip install "$wheel"
native/bin/python -m pip check
export JAX_PLATFORMS=cpu
native/bin/mjax check-env
native/bin/python -I -c 'import malthusjax; print(malthusjax.__file__)'
```

The import path should point to `native`'s `site-packages`.

## Run a Sphere optimization

Run one seed for two generations with the native genetic algorithm.
`native_sphere` evaluates the sum of squares using JAX. The `sphere` and
`bbob` catalog presets use EvoSAX for seeded BBOB instances.

```bash
native/bin/python -I - <<'PY'
import math

from malthusjax.composer import Composer

result = Composer.create_default().quick_run(
    fitness="native_sphere",
    genome_length=4,
    pop_size=16,
    generations=2,
    seeds=(42,),
    output_dir="results",
)
run = result.runs[0]
assert run.status == "success", run.error
assert math.isfinite(run.metrics["best_fitness"]), run.metrics
print(run.status, run.metrics["best_fitness"])
PY
```

## Run the same optimization from TOML

Save this configuration as `sphere.toml`.

```toml
[experiment]
name = "sphere_test"

[experiment.shared]
fitness = "native_sphere"
genome_length = 4
pop_size = 16
generations = 2
seeds = [42]

[pipelines.ga]
backend = "malthusjax"
```

```bash
native/bin/mjax run sphere.toml
```

The CLI writes the run record to `results/sphere/data/pipeline_ga/seed_42.json`.

Continue with [research workflows](research_workflows.md) or retrieve the
agent guide with `native/bin/mjax skill`.
