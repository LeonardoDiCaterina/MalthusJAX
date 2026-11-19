"""Recover verified Pareto and archive routes from ordinary runner results."""

import itertools
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import struct

from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import EngineOutputOptions, build_mo_engine
from malthusjax.composer.adapters.map_elites_adapter import MapElitesEngineAdapter
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.operators.crossover import OrderCrossover
from malthusjax.operators.emitters.genetic import GeneticCrossoverEmitter, GeneticMutationEmitter
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.mutation.categorical import SwapMutation

# Stations A–G and depot H. Two urgency clusters lie across a river at x=4.
POINTS = ((1, 2), (1, 7), (3, 5), (6, 7), (6, 1), (9, 3), (9, 9), (1, 0))
PRIORITY = (1, 3, 2, 2, 8, 5, 1)
DEPOT = 7
POP = 24
GENERATIONS = 24
GRID = tuple(itertools.product((0.0, 0.5, 1.0), repeat=2))
CONFIG = CategoricalGenomeConfig(num_categories=7, shape=(7,), is_permutation=True)


def jax_metrics(route):
    coords = jnp.asarray(POINTS)
    trip = jnp.concatenate((jnp.array([DEPOT]), route, jnp.array([DEPOT])))
    stops = coords[trip]
    legs = jnp.sum(jnp.abs(stops[1:] - stops[:-1]), axis=1)
    weighted_arrivals = jnp.dot(jnp.cumsum(legs[:-1]), jnp.asarray(PRIORITY)[route])
    crossings = jnp.sum((stops[:-1, 0] < 4) != (stops[1:, 0] < 4))
    urgent_burden = jnp.sum(jnp.asarray(PRIORITY)[route[:2]])
    return jnp.sum(legs), weighted_arrivals, crossings, urgent_burden


def exact_metrics(route):
    """Independent standard-library oracle."""
    assert tuple(sorted(route)) == tuple(range(7)), route
    time = latency = crossings = 0
    previous = DEPOT
    for station in (*route, DEPOT):
        x, y = POINTS[previous]
        a, b = POINTS[station]
        time += abs(x - a) + abs(y - b)
        crossings += (x < 4) != (a < 4)
        if station != DEPOT:
            latency += time * PRIORITY[station]
        previous = station
    return time, latency, crossings, sum(PRIORITY[i] for i in route[:2])


@struct.dataclass
class Costs(BaseMOEvaluator):
    def evaluate(self, genome: CategoricalGenome):
        time, latency, *_ = jax_metrics(genome.values)
        return jnp.array([time, latency], dtype=jnp.float32)


@struct.dataclass
class Layout(BaseQDEvaluator):
    def evaluate_qd(self, genome: CategoricalGenome):
        time, _, crossings, urgent = jax_metrics(genome.values)
        return time.astype(jnp.float32), jnp.array([(crossings - 2) / 6, (urgent - 2) / 11])


def emitter():
    return GeneticMutationEmitter(
        mutation=SwapMutation(mutation_rate=1.0), genome_config=CONFIG, _batch_size=POP
    )


def main():
    evaluator_config = BaseEvaluatorConfig(maximize=False)
    mo = build_mo_engine(
        Costs(config=evaluator_config, data=None),
        emitter(),
        genome_type="categorical",
        genome_config=CONFIG,
        pop_size=POP,
        generations=GENERATIONS,
    )
    qd = MapElitesEngineAdapter(
        MapElitesEngine(
            emitter=MixingEmitter(
                emitter().replace(_batch_size=POP // 2),
                GeneticCrossoverEmitter(
                    crossover=OrderCrossover(),
                    genome_config=CONFIG,
                    _batch_size=POP // 2,
                ),
            ),
            evaluator=Layout(config=evaluator_config, data=None),
            engine_params=MapElitesEngineParams(
                pop_size=POP, num_generations=GENERATIONS, maximize=False
            ),
        ),
        POP,
        False,
        None,
        centroids=jnp.array(GRID),
    )

    for name, engine in (("pareto", mo), ("archive", qd)):
        options = EngineOutputOptions(
            export_pareto=name == "pareto", export_archive=name == "archive"
        )
        run = (
            BenchmarkRunner(
                engine, output_dir=Path("permutation-results") / name, engine_outputs=options
            )
            .run([22])
            .runs[0]
        )
        if run.status != "success":
            raise RuntimeError(run.error)
        with np.load(run.artifacts[name], allow_pickle=False) as designs:
            for index, route in enumerate(designs["genotypes"]):
                length, latency, *_ = exact_metrics(route.tolist())
                if name == "pareto":
                    np.testing.assert_allclose(designs["objectives"][index], [length, latency])
                else:
                    np.testing.assert_allclose(designs["archive_fitness"][index], -length)
            print(f"{name}: verified {len(designs['genotypes'])} routes in {run.artifacts[name]}")


if __name__ == "__main__":
    main()
