import time
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
        # Dummy fitness: Rastrigin-like function (negative for maximization)
        x = population.genes.values
        A = 10.0
        n = x.shape[1]
        
        # Fitness
        fitnesses = -(A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis=1))
        
        # Descriptors: just take the first two dimensions of the genome
        # Map them from [-5.12, 5.12] to [0, 1]
        desc_x = (x[:, 0] + 5.12) / 10.24
        desc_y = (x[:, 1] + 5.12) / 10.24
        descriptors = jnp.stack([desc_x, desc_y], axis=-1)
        
        return population.replace(
            fitness=fitnesses,
            info={"descriptors": descriptors}
        )

def main():
    print("Initializing Standard MalthusJAX Components...")
    genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.12, 5.12))
    
    # 1. Operators
    # We set up a decaying mutation rate to prove the generation parameter passes through correctly
    mutation = GaussianMutation(
        mutation_strength=1.0,
        mutation_rate=0.5
    )
    crossover = UniformCrossover(
        crossover_rate=1.0
    )
    
    print("Building MalthusJAX Compositional Emitter...")
    # Half batch mutation, half batch crossover
    mut_emitter = GeneticMutationEmitter(_batch_size=32, mutation=mutation, genome_config=genome_config)
    cx_emitter = GeneticCrossoverEmitter(_batch_size=32, crossover=crossover, genome_config=genome_config)
    emitter = MixingEmitter(emitter_a=mut_emitter, emitter_b=cx_emitter)
    
    # 2. Setup Evaluator & Engine
    evaluator = DummyRealEvaluator(config=None, data=None)
    
    engine_params = MapElitesEngineParams(
        pop_size=64, # Total batch size
        num_generations=500
    )
    
    engine = MapElitesEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    print("Computing QDAX CVT Centroids...")
    centroids = compute_cvt_centroids(
        num_descriptors=2, 
        num_init_cvt_samples=5000, 
        num_centroids=100, 
        minval=0.0, 
        maxval=1.0, 
        key=k1
    )
    
    # 3. Initial Population
    initial_pop = genome_config.init_population(k2, engine_params.pop_size)
    
    print("Initializing MAP-Elites Engine...")
    t0 = time.time()
    engine_state = engine.init_state(k3, initial_pop, centroids=centroids)
    print(f"Calling engine.init_state... Done in {time.time()-t0:.3f}s")
    
    print("\n" + "="*40)
    print("MALTHUSJAX RNG ALLOCATION SUMMARY")
    print("="*40)
    
    # Analyze the Emitter Tree
    def _print_emitter_allocation(em, depth=0, indent="  "):
        name = em.__class__.__name__
        keys = em.num_keys()
        prefix = indent * depth
        if hasattr(em, "emitter_a"):
            print(f"{prefix}[+] {name}: requesting {keys} total keys. Routing to:")
            _print_emitter_allocation(em.emitter_a, depth + 1, indent)
            _print_emitter_allocation(em.emitter_b, depth + 1, indent)
        else:
            sampling = em.num_keys_for_sampling()
            atomic = keys - sampling
            print(f"{prefix}[-] {name}: {keys} keys (Sampling: {sampling}, Atomic: {atomic})")
            
    _print_emitter_allocation(emitter)
    
    total_engine_budget = emitter.num_keys() + 3
    print("-" * 40)
    print(f"Total Emitter Budget: {emitter.num_keys()} keys")
    print(f"Engine Fixed Budget:  3 keys (Eval, Tell, NextGen)")
    print(f"TOTAL RNG BUDGET:     {total_engine_budget} keys derived perfectly in parallel via jax.random.fold_in")
    print("="*40 + "\n")
    
    print(f"Running Compositional QD Evolution for {engine_params.num_generations} generations...")
    final_state, history, elapsed = engine.run(engine_state, time_it=True, compile=True, verbose=True)
    
    print(f"\nEvolution Complete in {elapsed:.2f}s!")
    print(f"Final Best Fitness: {float(final_state.best_fitness):.2f}")
    print(f"Final QD Score: {history.qd_score[-1]:.2f}")
    print(f"Final Map Coverage: {history.coverage[-1] * 100:.2f}% ({jnp.sum(final_state.repertoire.fitnesses > -jnp.inf)} cells)")

if __name__ == "__main__":
    main()
