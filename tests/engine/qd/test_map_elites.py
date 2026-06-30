import pytest
import jax
import jax.numpy as jnp

from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.core.fitness.base import BaseQDEvaluator
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter, GeneticCrossoverEmitter
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

class DummyRealEvaluator(BaseQDEvaluator):
    def evaluate_population(self, population: RealPopulation) -> RealPopulation:
        x = population.genes.values
        A = 10.0
        n = x.shape[1]
        
        fitnesses = -(A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis=1))
        
        desc_x = (x[:, 0] + 5.12) / 10.24
        desc_y = (x[:, 1] + 5.12) / 10.24
        descriptors = jnp.stack([desc_x, desc_y], axis=-1)
        
        return population.replace(
            fitness=fitnesses,
            info={"descriptors": descriptors}
        )

def test_map_elites_engine_fold_in():
    genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.12, 5.12))
    
    mutation = GaussianMutation(mutation_strength=1.0, mutation_rate=0.5)
    crossover = UniformCrossover(crossover_rate=1.0)
    
    mut_emitter = GeneticMutationEmitter(_batch_size=8, mutation=mutation, genome_config=genome_config)
    cx_emitter = GeneticCrossoverEmitter(_batch_size=8, crossover=crossover, genome_config=genome_config)
    emitter = MixingEmitter(emitter_a=mut_emitter, emitter_b=cx_emitter)
    
    evaluator = DummyRealEvaluator(config=None, data=None)
    
    engine_params = MapElitesEngineParams(
        pop_size=16,
        num_generations=10,
        key_derivation="fold_in"
    )
    
    engine = MapElitesEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    centroids = compute_cvt_centroids(
        num_descriptors=2, 
        num_init_cvt_samples=1000, 
        num_centroids=10, 
        minval=0.0, 
        maxval=1.0, 
        key=k1
    )
    
    initial_pop = genome_config.init_population(k2, engine_params.pop_size)
    engine_state = engine.init_state(k3, initial_pop, centroids=centroids)
    
    final_state, history, _ = engine.run(engine_state, compile=True)
    
    # Assert coverage increased over time or is correctly recorded
    assert history.coverage[-1] > 0
    assert history.qd_score[-1] < 0 # Rastrigin yields negative fitness
    assert jnp.sum(final_state.repertoire.fitnesses > -jnp.inf) > 0

def test_map_elites_engine_split():
    # Test split fallback
    genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.12, 5.12))
    mutation = GaussianMutation(mutation_strength=1.0, mutation_rate=0.5)
    emitter = GeneticMutationEmitter(_batch_size=16, mutation=mutation, genome_config=genome_config)
    
    evaluator = DummyRealEvaluator(config=None, data=None)
    
    engine_params = MapElitesEngineParams(
        pop_size=16,
        num_generations=2,
        key_derivation="split"
    )
    
    engine = MapElitesEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    centroids = compute_cvt_centroids(2, 1000, 10, 0.0, 1.0, k1)
    initial_pop = genome_config.init_population(k2, engine_params.pop_size)
    engine_state = engine.init_state(k3, initial_pop, centroids=centroids)
    
    final_state, history, _ = engine.run(engine_state, compile=True)
    assert final_state.generation == 2
