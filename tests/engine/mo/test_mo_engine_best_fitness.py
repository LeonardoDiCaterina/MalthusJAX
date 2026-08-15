import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.operators.mutation.real import GaussianMutation


@struct.dataclass
class MinMOConfig(BaseEvaluatorConfig):
    maximize: bool = struct.field(pytree_node=False, default=False)


@struct.dataclass
class MinMOEvaluator(BaseMOEvaluator[RealGenome, MinMOConfig, None]):
    """Evaluates genome directly. For minimization, lower values are better."""

    def evaluate(self, genome: RealGenome) -> jnp.ndarray:
        # Objective 1 is just the first gene, objective 2 is the second gene.
        return jnp.stack([genome.values[0], genome.values[1]], axis=-1)


def test_mo_engine_best_fitness_minimize():
    """
    Test that best_fitness correctly handles minimization (maximize=False).
    Under minimize, best_fitness should be the minimum objective value,
    and it should correspond exactly to best_genome's fitness.
    """
    key = jax.random.PRNGKey(42)
    pop_size = 10
    
    genome_config = RealGenomeConfig(shape=(2,), bounds=(0.0, 10.0))
    k_init, k_engine = jax.random.split(key)
    
    # We force specific values so we know exactly what the min and max are
    # pop[0] will be (1.0, 9.0) -> pareto front, obj0=1.0
    # pop[1] will be (2.0, 8.0) -> pareto front, obj0=2.0
    # pop[2] will be (9.0, 1.0) -> pareto front, obj0=9.0
    # ...
    values = jnp.array([
        [1.0, 9.0],
        [2.0, 8.0],
        [9.0, 1.0],
        [8.0, 2.0],
        [5.0, 5.0],
        [4.0, 6.0],
        [6.0, 4.0],
        [3.0, 7.0],
        [7.0, 3.0],
        [10.0, 10.0]  # Dominated
    ])
    
    initial_pop = genome_config.init_population(k_init, pop_size)
    # Inject our synthetic values
    initial_pop = initial_pop.replace(genes=initial_pop.genes.replace(values=values))

    evaluator = MinMOEvaluator(config=MinMOConfig(maximize=False), data=None)
    
    # Setup dummy emitter that just mutates slightly (or does nothing)
    mutation = GaussianMutation(mutation_rate=0.0) # 0 mutation, keep values same
    emitter = GeneticMutationEmitter(mutation=mutation, genome_config=genome_config, _batch_size=pop_size)

    engine_params = MOEngineParams(pop_size=pop_size, num_generations=1)
    engine = MOEngine(emitter=emitter, evaluator=evaluator, engine_params=engine_params)

    # Test init_state
    state = engine.init_state(k_engine, initial_pop)
    
    actual_best_fitness = state.best_fitness
    actual_best_genome = state.best_genome
    
    # Consistency check: best_fitness == evaluator.evaluate(best_genome)[0]
    expected_obj0_from_genome = evaluator.evaluate(actual_best_genome)[0]
    assert jnp.isclose(actual_best_fitness, expected_obj0_from_genome)
    
    # Regression check: it shouldn't be the max!
    assert actual_best_fitness < 9.9, f"best_fitness was max ({actual_best_fitness}), expected something from pareto front!"
    
    # Step once
    final_state, kpi = engine.step(state)
    
    actual_best_fitness_step = final_state.best_fitness
    actual_best_genome_step = final_state.best_genome
    
    expected_obj0_from_genome_step = evaluator.evaluate(actual_best_genome_step)[0]
    assert jnp.isclose(actual_best_fitness_step, expected_obj0_from_genome_step)
    
    # Kpi matches
    assert jnp.isclose(kpi.best_fitness, actual_best_fitness_step)
