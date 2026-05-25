import types
from types import SimpleNamespace

import numpy as np
import pytest

from examples.toy_gap_convergence import _shared_initial_population
from malthusjax.composer.composer import Composer
from malthusjax.benchmarking.results import ExperimentResult
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.composer.evosax_adapter import build_evosax_engine
import jax.random as jr
import jax


def _make_stub_quick_run(self, **kwargs):
    # Return a minimal ExperimentResult without executing heavy runs.
    return ExperimentResult(name=kwargs.get("experiment_name", "stub"), runs=[])


def test_compare_shared_initial_population_matches_toy(monkeypatch):
    args = SimpleNamespace(pop_size=12, dimensions=3, seed=0, function="rosenbrock")

    # Toy initial population
    toy_pop = _shared_initial_population(args)

    # Patch Composer.quick_run to avoid executing benchmarks
    monkeypatch.setattr(Composer, "quick_run", _make_stub_quick_run, raising=False)

    composer = Composer.create_default()

    pipelines = {
        "m_j": {"fitness": "bbob:fn_name=rosenbrock,num_dims=3,seed=0,maximize=false"},
        "ev": {"backend": "evosax", "fitness": "bbob:fn_name=rosenbrock,num_dims=3,seed=0,maximize=false"},
    }

    comparison = composer.compare(
        pipelines=pipelines,
        seeds=(0,),
        shared_initial_population=True,
        pop_seed=0,
        pop_size=args.pop_size,
        generations=1,
    )

    assert comparison.initial_population is not None
    np.testing.assert_allclose(np.asarray(comparison.initial_population), np.asarray(toy_pop))


@pytest.mark.parametrize("fn_name,dimensions,pop_size,elitism", [
    ("sphere", 2, 4, 0),
    ("rosenbrock", 3, 8, 0),
    ("rastrigin", 3, 8, 1),
    ("sphere", 5, 8, 2),
])
def test_end_to_end_equivalence_manual_vs_composer_vs_toml(tmp_path, fn_name, dimensions, pop_size, elitism):
    # Small quick experiment parameters
    generations = 1
    seed = 0
    elite_k = max(1, pop_size // 2) if elitism > 0 else 1

    # Build evaluator and shared initial population (same logic composer uses)
    evaluator = BBOBEvaluator.create(BBOBConfig(fn_name=fn_name, num_dims=dimensions, seed=seed, maximize=False))

    pop_key = jr.PRNGKey(seed)
    sample_keys = jr.split(pop_key, pop_size)
    shared_pop = jax.vmap(evaluator.evosax_problem.sample)(sample_keys)

    # --- Manual engine run (MalthusJAX) ---------------------------------
    genome_config = RealGenomeConfig(shape=(dimensions,), bounds=(-5.0, 5.0))
    engine_params = GeneticEngineParams(pop_size=pop_size, elitism=elitism, num_generations=generations)
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=pop_size, elite_k=elite_k),
        crossover=EvosaxUniformCrossoverWrapper(num_offspring=1, crossover_rate=0.5),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=0.1),
        enable_progress_bar=False,
    )

    initial_population = RealPopulation.from_array(shared_pop, genome_config, axis=0)
    evaluated_population = evaluator.evaluate_population(initial_population)
    # prepare state with injected population
    state = engine.init_state(jr.PRNGKey(seed)).replace(
        population=evaluated_population,
        best_genome=evaluated_population.genes[int(evaluated_population.fitness.argmin())],
        best_fitness=evaluated_population.fitness[int(evaluated_population.fitness.argmin())],
    )

    final_state, scan_history, _ = engine.run(state, compile=False, time_it=False)
    manual_best = float(final_state.best_fitness)

    # --- Composer.compare run --------------------------------------------
    composer = Composer.create_default()
    pipelines = {
        "malthus_manual": {
            "fitness": f"bbob:fn_name={fn_name},num_dims={dimensions},seed={seed},maximize=false",
            "pop_size": pop_size,
            "generations": generations,
            # Match manual engine operators and params
            "selection": f"elite_pool:elite_k={elite_k}",
            "crossover": "evosax_uniform_crossover:crossover_rate=0.5",
            "mutation": "evosax_gaussian:mutation_strength=0.1",
            "elitism": elitism,
        }
    }

    comparison = composer.compare(
        pipelines=pipelines,
        seeds=(seed,),
        shared_initial_population=True,
        pop_seed=seed,
        pop_size=pop_size,
        genome_length=dimensions,
        generations=generations,
    )

    comp_best = float(comparison.pipelines["malthus_manual"].runs[0].metrics["best_fitness"])

    # --- Composer.from_toml run -----------------------------------------
    # Provide matching operators in the TOML so Composer builds the same engine
    toml_content = f"""
[experiment.shared]
fitness = "bbob:fn_name={fn_name},num_dims={dimensions},seed={seed},maximize=false"
pop_size = {pop_size}
generations = {generations}
seeds = [{seed}]
genome_length = {dimensions}

[pipelines.simple]
selection = "elite_pool:elite_k={elite_k}"
crossover = "evosax_uniform_crossover:crossover_rate=0.5"
mutation = "evosax_gaussian:mutation_strength=0.1"
elitism = {elitism}
"""
    toml_path = tmp_path / "mini.toml"
    toml_path.write_text(toml_content)

    comparison_toml = Composer.from_toml(str(toml_path), shared_initial_population=True, pop_seed=seed)
    toml_best = float(comparison_toml.pipelines["simple"].runs[0].metrics["best_fitness"])

    # Compare all three
    np.testing.assert_allclose(manual_best, comp_best)
    np.testing.assert_allclose(manual_best, toml_best)


@pytest.mark.parametrize("fn_name,dimensions,pop_size,seed", [
    ("sphere", 2, 4, 0),
    ("rosenbrock", 3, 8, 0),
    ("rastrigin", 3, 8, 1),
    ("sphere", 5, 8, 2),
])
def test_evosax_backend_equivalence(tmp_path, fn_name, dimensions, pop_size, seed):
    generations = 1

    evaluator = BBOBEvaluator.create(BBOBConfig(fn_name=fn_name, num_dims=dimensions, seed=seed, maximize=False))

    pop_key = jr.PRNGKey(seed)
    sample_keys = jr.split(pop_key, pop_size)
    shared_pop = jax.vmap(evaluator.evosax_problem.sample)(sample_keys)

    # Manual evosax adapter run using the same initial population
    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=pop_size,
        generations=generations,
        initial_population=shared_pop,
        seed=seed,
    )

    manual_res = adapter.run_once(jr.PRNGKey(seed), compile=False)
    manual_best = float(manual_res["summary"]["best_fitness"])

    # Composer.compare with backend evosax
    composer = Composer.create_default()
    pipelines = {"evosax_manual": {"backend": "evosax", "evosax_strategy": "SimpleGA", "pop_size": pop_size, "generations": generations, "fitness": f"bbob:fn_name={fn_name},num_dims={dimensions},seed={seed},maximize=false"}}

    comparison = composer.compare(
        pipelines=pipelines,
        seeds=(seed,),
        shared_initial_population=True,
        pop_seed=seed,
        pop_size=pop_size,
        genome_length=dimensions,
        generations=generations,
    )

    comp_best = float(comparison.pipelines["evosax_manual"].runs[0].metrics["best_fitness"])

    # Composer.from_toml
    toml_content = f"""
[experiment.shared]
fitness = "bbob:fn_name={fn_name},num_dims={dimensions},seed={seed},maximize=false"
pop_size = {pop_size}
generations = {generations}
seeds = [{seed}]
genome_length = {dimensions}

[pipelines.evosax]
backend = "evosax"
evosax_strategy = "SimpleGA"
"""
    toml_path = tmp_path / "evosax_mini.toml"
    toml_path.write_text(toml_content)

    comparison_toml = Composer.from_toml(str(toml_path), shared_initial_population=True, pop_seed=seed)
    toml_best = float(comparison_toml.pipelines["evosax"].runs[0].metrics["best_fitness"])

    np.testing.assert_allclose(manual_best, comp_best)
    np.testing.assert_allclose(manual_best, toml_best)


def test_from_toml_shared_initial_population_matches_toy(tmp_path, monkeypatch):
    # Create minimal TOML experiment describing a single pipeline using BBOB
    toml_content = """
[experiment.shared]
fitness = "bbob:fn_name=rosenbrock,num_dims=3,seed=0,maximize=false"
pop_size = 12
generations = 1
seeds = [0]
# Provide genome_length to avoid OperatorCatalog parsing in tests
genome_length = 3

[pipelines.simple]
crossover = "blend:alpha=0.5"
"""

    toml_path = tmp_path / "minimal_experiment.toml"
    toml_path.write_text(toml_content)

    args = SimpleNamespace(pop_size=12, dimensions=3, seed=0, function="rosenbrock")
    toy_pop = _shared_initial_population(args)

    # Patch Composer.quick_run to avoid heavy runs
    monkeypatch.setattr(Composer, "quick_run", _make_stub_quick_run, raising=False)

    comparison = Composer.from_toml(str(toml_path), shared_initial_population=True, pop_seed=0)

    assert comparison.initial_population is not None
    np.testing.assert_allclose(np.asarray(comparison.initial_population), np.asarray(toy_pop))
