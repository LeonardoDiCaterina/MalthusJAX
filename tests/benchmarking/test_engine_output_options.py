"""Output options must preserve objectives and skip disabled extraction."""

import csv
import json
from dataclasses import replace

import pytest

from malthusjax.benchmarking.cli import _load_comparison, main
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import Composer, EngineOutputOptions
from malthusjax.composer.engine_factory import EngineFactory
from malthusjax.composer.experiment_config import ExperimentConfig
from malthusjax.core.timing_records import TimingPolicy
from malthusjax.engine.schedules import TrackBest


@pytest.mark.parametrize("mode", [TrackBest.LIGHT, TrackBest.FULL, TrackBest.NONE])
def test_native_solution_score_matches_its_genome(mode):
    result = Composer.create_default().quick_run(
        fitness="native_sphere",
        genome_length=2,
        pop_size=8,
        generations=4,
        seeds=(1, 2, 3),
        mutation="gaussian:mutation_rate=1.0,mutation_strength=3.0",
        elitism=0,
        track_best=mode,
        write_artifacts=False,
    )
    for run in result.runs:
        assert run.status == "success", run.error
        solution = run.solution
        assert solution["fitness"] == pytest.approx(sum(gene**2 for gene in solution["value"]))
        if mode == TrackBest.FULL:
            assert solution["selection_scope"] == "best_seen"
            assert solution["fitness"] == run.metrics["best_fitness"]
        else:
            assert solution["selection_scope"] == "final_population_best"
        if mode == TrackBest.LIGHT:
            assert solution["fitness"] > run.metrics["best_fitness"]


@pytest.mark.parametrize(
    ("backend", "python_loop", "extra_metrics"),
    [
        ("malthusjax", False, ()),
        ("map_elites", False, ("coverage", "qd_score")),
        ("mo", False, ("num_pareto_optimal",)),
        ("evosax", False, ()),
        ("evosax", True, ()),
        ("qdax", False, ("coverage", "qd_score")),
        ("qdax", True, ("coverage", "qd_score")),
    ],
)
@pytest.mark.parametrize("maximize", [False, True])
def test_output_contract_preserves_terminal_metrics(backend, python_loop, extra_metrics, maximize):
    if backend in {"map_elites", "qdax"}:
        pytest.importorskip("qdax")
    if backend == "evosax":
        pytest.importorskip("evosax")

    if backend == "mo":
        from tests.composer.test_mo_adapter import make_mo_adapter

        engine = make_mo_adapter(maximize)
    else:
        options = {}
        if backend == "map_elites":
            import jax.numpy as jnp

            from malthusjax.composer.catalog import OperatorCatalog
            from malthusjax.composer.strategies.core import MapElitesStrategy
            from malthusjax.core.genome.real_genome import RealGenomeConfig
            from malthusjax.operators.emitters.genetic import GeneticMutationEmitter

            options["strategy"] = MapElitesStrategy(
                emitter=GeneticMutationEmitter(
                    mutation=OperatorCatalog().get("gaussian:mutation_rate=1.0"),
                    genome_config=RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0)),
                    _batch_size=8,
                ),
                centroids=jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
                num_centroids=3,
            )
        engine = EngineFactory.build(
            ExperimentConfig.from_quick_run_kwargs(
                backend=backend,
                fitness="native_sphere",
                genome_length=2,
                pop_size=8,
                generations=3,
                qdax_num_centroids=9,
                maximize=maximize,
                use_python_loop=python_loop,
                **options,
            )
        )

    def run(options):
        return BenchmarkRunner(engine, engine_outputs=options).run([1]).runs[0]

    full = run(EngineOutputOptions())
    reduced = run(EngineOutputOptions(collect_history=False, capture_solution=False))
    restored = run(EngineOutputOptions())
    for observation in (full, reduced, restored):
        assert observation.status == "success", observation.error
    assert len(full.history) == 3
    assert reduced.history == []
    assert reduced.solution is None
    assert "population_sha256" not in reduced.initialization
    assert restored.history == full.history
    for metric in ("best_fitness", "total_evaluations", "final_generation", *extra_metrics):
        assert reduced.metrics[metric] == pytest.approx(full.metrics[metric])
        assert restored.metrics[metric] == full.metrics[metric]
    assert "fitness_auc" not in reduced.metrics


def test_runner_dispatches_once_and_disables_repeat_extraction(tmp_path):
    calls = []

    class Engine:
        def run_with_options(self, key, options):
            calls.append(options)
            return {"summary": {"best_fitness": 1.0}}

    requested = EngineOutputOptions(capture_population_fingerprint=True, export_archive=True)
    run = (
        BenchmarkRunner(
            Engine(),
            output_dir=tmp_path,
            engine_outputs=requested,
            timing_policy=TimingPolicy(target_phase="prepared_execution_repeat", repetitions=1),
        )
        .run([1])
        .runs[0]
    )
    assert run.status == "success"
    assert calls == [
        requested,
        replace(
            requested,
            capture_solution=False,
            capture_population_fingerprint=False,
            export_archive=False,
        ),
    ]
    assert set(run.output_status["unavailable"]) == {
        "solution",
        "population_fingerprint",
        "archive",
    }

    class Broken:
        def run_with_options(self, key, options):
            raise TypeError("from inside the engine")

        def run_once(self, key):
            pytest.fail("A failed output-aware execution was retried")

    failed = BenchmarkRunner(Broken()).run([1]).runs[0]
    assert failed.status == "error"
    assert "from inside the engine" in failed.error


def test_output_config_round_trip_and_scope_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig.from_quick_run_kwargs(retain_history=False, capture_solution=False)
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    source = tmp_path / "options.toml"
    source.write_text("""
[experiment.shared]
fitness = "native_sphere"
genome_length = 2
pop_size = 8
generations = 2
seeds = [1]
retain_history = false

[pipelines.raw]
serialize_history = true
write_history_csv = false

[pipelines.csv]
retain_history = true

[pipelines.drop]
""")
    assert main(["run", str(source)]) == 0
    root = tmp_path / "results/options"
    with (root / "histories_combined.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["pipeline"] for row in rows} == {"csv"}
    result = _load_comparison(root)
    assert set(result.pipelines) == {"csv", "drop", "raw"}
    assert len(result.pipelines["raw"].runs[0].history) == 2
    assert result.pipelines["drop"].runs[0].history == []
    assert main(["analyze", str(root)]) == 0
    assert {path.stem for path in (root / "analysis").glob("*_summary.json")} == {
        "csv_summary",
        "drop_summary",
        "raw_summary",
    }
    assert (root / "exports/pipeline_csv/histories_combined.csv").is_file()
    for name in ("raw", "drop"):
        assert not (root / f"exports/pipeline_{name}/histories_combined.csv").exists()
        assert (root / f"exports/pipeline_{name}/timing_trials.csv").is_file()
        assert (root / f"data/pipeline_{name}/seed_1.json").is_file()


def test_base_engine_extension_inherits_static_history_mode():
    import jax.numpy as jnp

    from malthusjax.engine.base import AbstractEngine, AbstractEngineParams, _get_evolution_kernel

    class Minimal(AbstractEngine):
        def init_state(self, seed):
            return jnp.asarray(seed, dtype=jnp.float32)

        def step(self, state):
            state = state + 1
            return state, {"metric": state * 2}

    engine = Minimal(AbstractEngineParams(num_generations=3))
    full = _get_evolution_kernel(engine.engine_params, return_history=True)
    empty = _get_evolution_kernel(engine.engine_params, return_history=False)
    assert full is not empty
    assert full is _get_evolution_kernel(engine.engine_params, return_history=True)
    assert empty.lower(engine, engine.init_state(0)).out_info[1] == ()

    for collect in (True, False):
        prepared = engine.prepare_run(engine.init_state(0), return_history=collect)
        state, history, _ = engine.run(
            engine.init_state(0), return_history=collect, prepared_run=prepared
        )
        assert float(state) == 3.0
        if collect:
            assert history["metric"].tolist() == [2, 4, 6]
        else:
            assert history == ()
    with pytest.raises(ValueError, match="configuration"):
        engine.run(engine.init_state(0), return_history=True, prepared_run=prepared)


def test_legacy_extension_remains_callable():
    class Legacy:
        def __init__(self):
            self.calls = 0

        def run_once(self, key):
            self.calls += 1
            return {
                "summary": {"best_fitness": 1.0, "best_solution": [1.0]},
                "history": [{"generation": 1, "best_fitness": 1.0}],
            }

    engine = Legacy()
    run = (
        BenchmarkRunner(engine, engine_outputs=EngineOutputOptions(collect_history=False))
        .run([1])
        .runs[0]
    )
    assert engine.calls == 1
    assert run.history == []
    assert run.output_status["collection_control"] == "uncontrolled"
    assert run.output_status["collect_history"] is True
    assert run.solution["selection_scope"] == "unknown"
    assert "fitness" not in run.solution


@pytest.mark.parametrize(
    ("mode", "available"),
    [("none", ()), ("basic", ("mean_fitness",)), ("all", ("mean_fitness", "std_fitness"))],
)
def test_native_ga_does_not_report_dummy_metric_values(mode, available):
    run = (
        Composer.create_default()
        .quick_run(
            fitness="native_sphere",
            genome_length=2,
            pop_size=8,
            generations=2,
            seeds=(1,),
            track_metrics=mode,
            write_artifacts=False,
        )
        .runs[0]
    )
    assert run.status == "success", run.error
    metrics = {"mean_fitness", "std_fitness"}
    assert metrics & run.unavailable_metrics.keys() == metrics - set(available)
    assert all(metrics & row.keys() == set(available) for row in run.history)


@pytest.mark.parametrize(
    ("mode", "scope"),
    [
        (0, "final_population_best"),
        (1, "final_population_best"),
        (2, "best_seen"),
    ],
)
def test_cli_solution_payload_round_trips(tmp_path, monkeypatch, mode, scope):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "solution.toml"
    source.write_text(f"""
[experiment]
shared_initial_population = false

[experiment.shared]
fitness = "native_sphere"
genome_length = 2
pop_size = 8
generations = 4
elitism = 0
track_best = {mode}
seeds = [1]
mutation = "gaussian:mutation_rate=1.0,mutation_strength=3.0"

[pipelines.ga]
""")
    assert main(["run", str(source)]) == 0
    root = tmp_path / "results/solution"
    record = json.loads((root / "data/pipeline_ga/seed_1.json").read_text())
    assert record["status"] == "success", record["error"]
    solution = record["solution"]
    assert solution["selection_scope"] == scope
    assert solution["fitness"] == pytest.approx(sum(gene**2 for gene in solution["value"]))
    if mode == TrackBest.LIGHT:
        assert solution["fitness"] > record["metrics"]["best_fitness"]
    else:
        assert solution["fitness"] == record["metrics"]["best_fitness"]
    assert _load_comparison(root).pipelines["ga"].runs[0].solution == solution


def test_light_solution_uses_recorded_evaluation(tmp_path, monkeypatch):
    """A reevaluation can give a different score for the same candidate."""
    config = ExperimentConfig.from_quick_run_kwargs(
        fitness="native_sphere",
        genome_length=2,
        pop_size=8,
        generations=2,
    )
    engine = EngineFactory.build(config)
    original = type(engine.genetic_engine).run

    def with_observational_noise(self, *args, **kwargs):
        state, history, seconds = original(self, *args, **kwargs)
        noisy = state.population.replace(fitness=state.population.fitness + 17.0)
        return state.replace(population=noisy), history, seconds

    monkeypatch.setattr(type(engine.genetic_engine), "run", with_observational_noise)
    run = BenchmarkRunner(engine, output_dir=tmp_path).run([1]).runs[0]
    assert run.status == "success", run.error
    clean_score = sum(gene**2 for gene in run.solution["value"])
    assert run.solution["fitness"] == pytest.approx(clean_score + 17.0)


def test_native_disabled_extraction_skips_transfer_and_history_writes(tmp_path, monkeypatch):
    from jax import flatten_util

    import malthusjax.benchmarking.io as artifact_io
    import malthusjax.composer.engine_factory as native

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled host output work was invoked")

    monkeypatch.setattr(native, "population_fingerprint", forbidden)
    monkeypatch.setattr(flatten_util, "ravel_pytree", forbidden)
    monkeypatch.setattr(artifact_io, "write_histories_csv", forbidden)
    result = Composer.create_default().quick_run(
        fitness="native_sphere",
        genome_length=2,
        pop_size=8,
        generations=2,
        seeds=(1,),
        capture_solution=False,
        write_history_csv=False,
        output_dir=tmp_path,
    )
    assert result.runs[0].status == "success", result.runs[0].error
    assert result.runs[0].solution is None
    assert len(result.runs[0].history) == 2
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "timing_trials.csv").is_file()


@pytest.mark.parametrize("logging", [False, True])
def test_optional_ga_reductions_are_absent_from_lowered_modes(logging):
    import jax

    from malthusjax.engine.base import _get_evolution_kernel

    operations = {}
    for mode in ("none", "basic", "all"):
        engine = EngineFactory.build(
            ExperimentConfig.from_quick_run_kwargs(
                fitness="native_sphere",
                genome_length=2,
                pop_size=8,
                generations=2,
                track_metrics=mode,
                log_interval=1 if logging else None,
            )
        ).genetic_engine
        state = engine.init_state(jax.random.key(12))
        kernel = _get_evolution_kernel(engine.engine_params)
        lowered = kernel.lower(engine, state)
        ir = str(lowered.compiler_ir("stablehlo"))
        operations[mode] = (ir.count("stablehlo.reduce("), ir.count("stablehlo.sqrt"))

    # Compare the operations added by each mode with the same optimizer and
    # logging configuration. Selection/evaluation have their own reductions.
    assert operations["none"][0] < operations["basic"][0] < operations["all"][0]
    assert operations["none"][1] == operations["basic"][1] < operations["all"][1]
