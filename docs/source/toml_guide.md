# TOML experiments

Start with the [Sphere configuration](quickstart.md) in a scratch directory.

Put common settings in `[experiment.shared]` and overrides in each
`[pipelines.NAME]` table. Set `shared_initial_population = false` under
`[experiment]` to let the backend initialize per optimizer seed.

Run `mjax run CONFIG` or `mjax parity CONFIG`. Both publish a results
directory. Select a fresh destination with `--output-dir` when repeating a
run. Inspect `metadata/run_manifest.json` and `data/pipeline_*/seed_*.json`.
The `stats` extra enables paired analysis with `mjax analyze RESULTS`.

See [research workflows](research_workflows.md) for a two-pipeline example.
