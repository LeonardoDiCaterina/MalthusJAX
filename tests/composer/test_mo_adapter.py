"""Tests for the Composer MOEngineAdapter."""

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.composer.mo_factory import build_mo_engine
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.binary_genome import BinaryGenome
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.mutation.binary import BitFlipMutation


@struct.dataclass
class DummyMOConfig(BaseEvaluatorConfig):
    maximize: bool = struct.field(pytree_node=False, default=True)


@struct.dataclass
class DummyMOEvaluator(BaseMOEvaluator[BinaryGenome, DummyMOConfig, None]):
    def evaluate(self, genome: BinaryGenome) -> jnp.ndarray:
        # 2 Objectives: Sum of first half, Sum of second half
        mid = genome.values.shape[-1] // 2
        obj1 = jnp.sum(genome.values[:mid])
        obj2 = jnp.sum(genome.values[mid:])
        return jnp.stack([obj1, obj2], axis=-1)


from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.operators.emitters.genetic import GeneticCrossoverEmitter, GeneticMutationEmitter


def test_build_mo_engine_and_run():
    """Verify that build_mo_engine creates a working adapter that can run_once."""

    # 1. Instantiate the evaluator
    eval_config = DummyMOConfig(maximize=True)
    evaluator = DummyMOEvaluator(config=eval_config, data=None)

    # 2. Instantiate the emitter
    genome_config = BinaryGenomeConfig(length=10)
    emitter = MixingEmitter(
        emitter_a=GeneticMutationEmitter(
            mutation=BitFlipMutation(mutation_rate=0.1), genome_config=genome_config, _batch_size=10
        ),
        emitter_b=GeneticCrossoverEmitter(
            crossover=UniformCrossover(crossover_rate=1.0),
            genome_config=genome_config,
            _batch_size=10,
        ),
    )

    # 3. Build the adapter via the composer factory
    adapter = build_mo_engine(
        fitness_evaluator=evaluator,
        emitter=emitter,
        genome_type="binary",
        pop_size=20,
        generations=3,
        genome_shape=(10,),
    )

    # 4. Run it
    key = jax.random.PRNGKey(42)
    results = adapter.run_once(key)

    # 5. Verify results format matches BenchmarkRunner.Engine protocol
    assert "history" in results
    assert "summary" in results
    assert "timings" in results

    assert len(results["history"]) == 3
    assert results["summary"]["final_generation"] == 3
    assert "num_pareto_optimal" in results["summary"]
    assert "max_crowding_distance" in results["history"][-1]
