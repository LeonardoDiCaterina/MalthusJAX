# MalthusJAX

[![JAX](https://img.shields.io/badge/JAX-0.11%E2%80%930.12-blue.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[Coverage scope and validation](docs/audit_remediation/P11.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MalthusJAX is a JAX framework for evolutionary computation. Run seeded
experiments, compare search methods, or reuse genomes, evaluators, and
operators in your own engines.

## First run on CPU

Follow the [wheel installation guide](docs/source/quickstart.md) to set up
Python 3.12. Then run:

```python
from malthusjax.composer import Composer

run = Composer.create_default().quick_run(
    fitness="native_sphere", genome_length=4, pop_size=16,
    generations=2, seeds=(42,), output_dir="results",
).runs[0]
assert run.status == "success", run.error
print(run.metrics["best_fitness"])
```

Continue with the [TOML quickstart](docs/source/quickstart.md), open the
[introductory notebook](notebooks/00_quickstart_in_5_minutes.ipynb), or browse
the [maintained examples](examples/README.md). Agents can retrieve the bundled
usage skill with `mjax skill`.

## API levels

| To do this | Start here |
| --- | --- |
| Run Python or TOML experiments and compare pipelines | [Composer](src/malthusjax/composer/README.md) |
| Reuse an evaluator across GA, QD, or multi-objective experiments | [Engines](src/malthusjax/engine/README.md) |
| Compose or implement genetic operators | [Operators](src/malthusjax/operators/README.md) |
| Define a genome, population, or evaluator | [Core](src/malthusjax/core/README.md) |

### Benchmark results

The full Phase 1 GPU study contains **64,800 runs** over 108 settings.
It compares direct upstream execution, the MalthusJAX adapter, and the native
mimic for EvoSAX SimpleGA and QDax MAP-Elites. The grid crosses six BBOB
functions, two instance seeds, dimensions 10/20/50, and generation budgets
50/150/500 at population 128, with 100 matched seeds per setting.

| Native vs. adapter | EvoSAX SimpleGA | QDax MAP-Elites |
|---|---:|---:|
| Pooled log₁₀ objective error difference | +0.00192 | −0.00502 |
| 90% CI lower bound | −0.00201 | −0.01210 |
| 90% CI upper bound | +0.00585 | +0.00206 |
| Settings meeting the ±0.05 equivalence bound | 85/108 | 53/108 |
| Geometric mean steady GPU speedup | 1.8175× | 0.9964× |
| 95% CI lower bound for speedup | 1.8169× | 0.9961× |
| 95% CI upper bound for speedup | 1.8180× | 0.9966× |

All six pooled objective contrasts met the equivalence bound. Adapter versus
direct upstream met it in all 108 settings for both libraries. QDax native
versus adapter archive coverage met the ±5 percentage point bound in all 108
settings. The frozen relative QD-score test was inconclusive.

Speedup is reference time divided by contender time, using the median of five
synchronized steady executions per run. Compilation and warm-up were measured
separately. The intervals use paired estimates clustered by seed.
[Study protocol](configs/publication/part1_confirmatory.json) ·
[Artifact specification](docs/part1_confirmatory_v2_artifact_spec.md)

> [!NOTE]
## Integrations and extensions

The [research workflows](docs/source/research_workflows.md) show tested CPU
paths and their required extras. See the [extension guide](docs/source/extension_cli_guide.md)
and [contributor setup](CONTRIBUTING.md) for plugins and development dependencies.

### Docker Deployment
MalthusJAX includes a robust, multi-stage `Dockerfile` to simplify deployment on shared GPU clusters. Isolate optional dependencies (`qdax`, `evosax`, `tensorneat`, `rl`) during the build:

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --build-arg EXTRAS="[cuda12,qdax,rl]" \
  -t malthusjax:latest .
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and extension tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
