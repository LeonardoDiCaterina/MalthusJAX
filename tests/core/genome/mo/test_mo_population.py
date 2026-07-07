"""Integration tests to prove MOPopulation handles survival and selection seamlessly."""

import pytest
import jax
import jax.numpy as jnp

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.binary_genome import BinaryGenome
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter, GeneticCrossoverEmitter
from malthusjax.core.genome.mo.population import MOPopulation
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig

# We use a dummy evaluation function for the integration test
def mock_multi_objective_eval(genome_values):
    # Just return 2 objectives for each genome based on sum of first half and second half
    # Maximize both
    mid = genome_values.shape[-1] // 2
    obj1 = jnp.sum(genome_values[:, :mid], axis=1)
    obj2 = jnp.sum(genome_values[:, mid:], axis=1)
    return jnp.stack([obj1, obj2], axis=-1)

def test_mo_population_with_mixing_emitter():
    """Test that existing MixingEmitter can ask() and tell() via MOPopulation."""
    
    pop_size = 10
    batch_size = 10
    genome_length = 8
    key = jax.random.PRNGKey(42)
    
    # 1. Setup Initial Population
    k_init, k_emitter, k_loop = jax.random.split(key, 3)
    init_genes = jax.random.bernoulli(k_init, p=0.5, shape=(pop_size, genome_length)).astype(jnp.int32)
    init_fitness = mock_multi_objective_eval(init_genes)
    
    initial_pop = BasePopulation(
        genes=BinaryGenome(values=init_genes),
        fitness=init_fitness,
        config=None,
        info=None
    )
    
    # 2. Upgrade to Smart MOPopulation
    mo_pop = MOPopulation.from_evaluated(initial_pop, maximize=True)
    
    # 3. Initialize MixingEmitter
    crossover = UniformCrossover(crossover_rate=1.0)
    mutation = BitFlipMutation(mutation_rate=0.1)
    genome_config = BinaryGenomeConfig(length=genome_length)
    emitter = MixingEmitter(
        emitter_a=GeneticMutationEmitter(
            mutation=mutation,
            genome_config=genome_config,
            _batch_size=batch_size // 2
        ),
        emitter_b=GeneticCrossoverEmitter(
            crossover=crossover,
            genome_config=genome_config,
            _batch_size=batch_size - (batch_size // 2)
        )
    )
    
    emitter_state = emitter.init(k_emitter, initial_pop)
    
    # 4. Perform one full mu+lambda cycle (ask -> eval -> merge -> truncate)
    
    # ask() calls mo_pop.select() internally using the provided key slice
    keys_for_ask = jax.random.split(k_loop, emitter.num_keys())
    offspring_pop, new_emitter_state = emitter.ask(emitter_state, mo_pop, keys_for_ask)
    
    assert offspring_pop.genes.values.shape == (batch_size, genome_length)
    
    # Evaluate offspring
    offspring_fitness = mock_multi_objective_eval(offspring_pop.genes.values)
    evaluated_offspring = offspring_pop.replace(fitness=offspring_fitness)
    
    # Survival Mechanism!
    # The MO Population merges the 10 elites + 10 offspring, and then truncates back down to 10
    surviving_pop = mo_pop.merge(evaluated_offspring).truncate(pop_size)
    
    # Verify population size was strictly maintained at 10 despite merging
    assert surviving_pop.fitness.shape == (pop_size, 2)
    assert surviving_pop.genes.values.shape == (pop_size, genome_length)
    
    # Verify ranks and crowding distance were computed
    assert surviving_pop.pareto_rank.shape == (pop_size,)
    assert surviving_pop.crowding_distance.shape == (pop_size,)
