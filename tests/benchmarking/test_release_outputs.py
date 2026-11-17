"""Release contracts for permutation inputs and native design outputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import EngineOutputOptions, build_mo_engine
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.initial_population import prepare_initial_population
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.operators.crossover import OrderCrossover
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.mutation.categorical import SwapMutation


@pytest.mark.parametrize("cuts", [(0, 0), (1, 3), (0, 5), (5, 5)])
def test_order_crossover_matches_reference(cuts):
    parents = ([0, 1, 2, 3, 4, 5], [2, 5, 0, 4, 1, 3])
    first, second = parents
    start, end = cuts
    retained = first[start : end + 1]
    cycle = second[end + 1 :] + second[: end + 1]
    fill = iter(value for value in cycle if value not in retained)
    expected = first.copy()
    for index in list(range(end + 1, len(first))) + list(range(start)):
        expected[index] = next(fill)

    config = CategoricalGenomeConfig(num_categories=6, shape=(6,), is_permutation=True)
    crossover = OperatorCatalog().get("order_crossover")
    assert isinstance(crossover, OrderCrossover)
    recombine = jax.jit(
        lambda a, b: crossover._recombine_one(
            CategoricalGenome(a),
            CategoricalGenome(b),
            (jnp.array(True), jnp.array(start), jnp.array(end)),
            config,
        ).values
    )
    np.testing.assert_array_equal(recombine(jnp.array(first), jnp.array(second)), expected)


def test_mixing_forwarding_and_permutation_input_validation():
    config = CategoricalGenomeConfig(num_categories=3, shape=(3,), is_permutation=True)
    emitter = GeneticMutationEmitter(mutation=SwapMutation(), genome_config=config, _batch_size=2)
    assert MixingEmitter(emitter, emitter).genome_config == config
    different = emitter.replace(genome_config=config.replace(num_categories=4, shape=(4,)))
    with pytest.raises(ValueError, match="compatible genome configurations"):
        _ = MixingEmitter(emitter, different).genome_config
    assert MixingEmitter(object(), emitter).genome_config == config

    genes = jnp.array([[1, 0, 2], [2, 0, 1]])
    population = prepare_initial_population(genes, config, 2)
    np.testing.assert_array_equal(population.genes.values, genes)
    assert np.issubdtype(population.genes.values.dtype, np.integer)
    np.testing.assert_array_equal(
        prepare_initial_population(population, config, 2).genes.values, genes
    )
    with pytest.raises(ValueError, match="complete permutation"):
        prepare_initial_population(jnp.zeros((2, 3), dtype=int), config, 2)
    with pytest.raises(ValueError, match="integer dtype"):
        prepare_initial_population(genes.astype(float), config, 2)


def test_categorical_mo_exports_recorded_pareto_objectives(tmp_path):
    from flax import struct

    from malthusjax.core.fitness.base import BaseEvaluatorConfig
    from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator

    @struct.dataclass
    class Placement(BaseMOEvaluator):
        def evaluate(self, genome):
            return genome.values[jnp.array([0, -1])].astype(jnp.float32)

    config = CategoricalGenomeConfig(num_categories=4, shape=(4,), is_permutation=True)
    emitter = GeneticMutationEmitter(
        mutation=SwapMutation(mutation_rate=1.0), genome_config=config, _batch_size=8
    )
    initial = config.init_population(jax.random.key(3), 8)
    engine = build_mo_engine(
        Placement(config=BaseEvaluatorConfig(maximize=True), data=None),
        emitter,
        genome_type="categorical",
        pop_size=8,
        generations=3,
        initial_population=initial,
    )
    run = (
        BenchmarkRunner(
            engine,
            output_dir=tmp_path,
            engine_outputs=EngineOutputOptions(export_pareto=True, collect_history=False),
        )
        .run([12])
        .runs[0]
    )
    assert run.status == "success", run.error
    assert initial.genes.values.shape == (8, 4)
    assert run.history == []
    with np.load(run.artifacts["pareto"], allow_pickle=False) as designs:
        routes = designs["genotypes"]
        objectives = designs["objectives"]
        assert len(routes) == run.metrics["num_pareto_optimal"] > 0
        np.testing.assert_array_equal(
            np.sort(routes, axis=1), np.tile(np.arange(4), (len(routes), 1))
        )
        np.testing.assert_array_equal(objectives, routes[:, [0, -1]])
        np.testing.assert_array_equal(designs["objective_direction"], ["maximize", "maximize"])
        for candidate in objectives:
            assert not np.any(
                np.all(objectives >= candidate, axis=1) & np.any(objectives > candidate, axis=1)
            )


def test_native_ga_uses_supplied_permutations_with_order_crossover():
    from flax import struct

    from malthusjax.composer.engine_factory import build_engine
    from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig

    @struct.dataclass
    class Placement(BaseEvaluator):
        def evaluate(self, genome):
            return (genome.values[0] + genome.values[-1]).astype(jnp.float32)

    config = CategoricalGenomeConfig(num_categories=4, shape=(4,), is_permutation=True)
    initial = config.init_population(jax.random.key(5), 8)
    expected = np.asarray(initial.genes.values).copy()
    engine = build_engine(
        Placement(config=BaseEvaluatorConfig(maximize=False), data=None),
        OperatorCatalog().get("tournament:num_selections=4,tournament_size=2"),
        OrderCrossover(crossover_rate=1.0),
        SwapMutation(mutation_rate=1.0),
        genome_config=config,
        initial_population=initial,
        genome_type="categorical",
        pop_size=8,
        generations=3,
    )
    result = BenchmarkRunner(engine).run([9]).runs[0]
    assert result.status == "success", result.error
    np.testing.assert_array_equal(initial.genes.values, expected)
    solution = result.solution
    assert sorted(solution["value"]) == list(range(4))
    assert solution["fitness"] == solution["value"][0] + solution["value"][-1]
