# Maintained examples

Build and install the wheel using the [quickstart](../docs/source/quickstart.md).
Copy an example directory into a scratch directory before running it.

| Example | Dependencies | Command and result |
| --- | --- | --- |
| [First Sphere run](../docs/source/quickstart.md) | Base wheel | `mjax run sphere.toml` writes seed JSON under `results/sphere`. |
| [Corridor sensor placement](corridor_sensors) | Base for native GA, separate `evosax` and `qdax` extras for their backends | `python run.py malthusjax` prints recovered coordinates. `mjax parity sensors.toml` writes pipeline results under `results/sensors`. |
| [Permutation inspection routes](permutation_routes/run.py) | `qdax` for native archive storage | `python run.py` verifies the exported Pareto and archive routes in `permutation-results`. |

## Result capabilities

| Tested path | Result access |
| --- | --- |
| Native GA | `run.solution` gives a genome, its own fitness, and selection scope. |
| Native MAP-Elites | `run.solution` represents the best occupied cell. Request `export_archive=True` for all occupied cells. |
| Native multi-objective | Request `export_pareto=True` for final rank-zero designs and objective vectors. Duplicate vectors are retained. |
| Ring island GA | `run.solution` gives the terminal winner across islands through the output-aware runner. |
| QDax adapter | Request `export_archive=True` for occupied archive cells. |

Pass collection options through `EngineOutputOptions` and file retention
options through `RunArtifactOptions` when using `BenchmarkRunner` directly.
A QD evaluator's `evaluate_qd` returns fitness and descriptors. Scalar engines
can use its `evaluate` interface. When you call prepared engines directly,
copy state or population buffers you intend to reuse after donation.

Other scripts in this directory include showcases, diagnostics, and
historical research runs. See [the framework guides](../docs/framework_guides/index.rst)
for extension APIs and the [benchmark results](../README.md#benchmark-results)
for the published study scope.
