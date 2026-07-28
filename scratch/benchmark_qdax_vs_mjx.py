import time
import jax
import jax.numpy as jnp
from functools import partial

# --- QDAX Imports ---
from qdax.core.map_elites import MAPElites as QDAX_MAPElites
from qdax.core.emitters.standard_emitters import MixingEmitter as QDAX_MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

# --- MalthusJAX Imports ---
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.core.fitness.base import BaseQDEvaluator
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter, GeneticCrossoverEmitter
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover

# --- Common Setup ---
BATCH_SIZE = 1024
NUM_GENS = 1000
NUM_VARS = 10
NUM_CENTROIDS = 100

def rastrigin_fitness(x):
    A = 10.0
    n = x.shape[1]
    return -(A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis=1))

def rastrigin_descriptors(x):
    desc_x = (x[:, 0] + 5.12) / 10.24
    desc_y = (x[:, 1] + 5.12) / 10.24
    return jnp.stack([desc_x, desc_y], axis=-1)

# QDAX Scoring Function
def qdax_scoring_function(x, random_key):
    fitnesses = rastrigin_fitness(x)
    descriptors = rastrigin_descriptors(x)
    return fitnesses, descriptors, {}

# MalthusJAX Evaluator
class MJX_Evaluator(BaseQDEvaluator):
    def evaluate_population(self, population: RealPopulation) -> RealPopulation:
        x = population.genes.values
        fitnesses = rastrigin_fitness(x)
        descriptors = rastrigin_descriptors(x)
        return population.replace(
            fitness=fitnesses,
            info={"descriptors": descriptors}
        )

def run_qdax(k):
    print("--- Running QDAX Benchmark ---")
    k1, k2, k3, k4 = jax.random.split(k, 4)
    
    # Emitter
    def dummy_mutation(x, key):
        return x + jax.random.normal(key, x.shape) * 1.0
        
    def dummy_crossover(x1, x2, key):
        mask = jax.random.bernoulli(key, p=0.5, shape=x1.shape)
        return jnp.where(mask, x1, x2)
        
    emitter = QDAX_MixingEmitter(
        mutation_fn=dummy_mutation,
        variation_fn=dummy_crossover,
        variation_percentage=0.5,
        batch_size=BATCH_SIZE
    )
    
    map_elites = QDAX_MAPElites(
        scoring_function=qdax_scoring_function,
        emitter=emitter,
        metrics_function=lambda r: {"qd_score": r.fitnesses.sum()}
    )
    
    centroids = compute_cvt_centroids(2, 5000, NUM_CENTROIDS, 0.0, 1.0, k1)
    
    # Init
    init_pop = jax.random.uniform(k2, shape=(BATCH_SIZE, NUM_VARS), minval=-5.12, maxval=5.12)
    
    t0 = time.time()
    repertoire, emitter_state, _ = map_elites.init(init_pop, centroids, k3)
    
    # Scan loop
    @jax.jit
    def step(carry, unused):
        rep, est, key = carry
        key, subkey = jax.random.split(key)
        rep, est, metrics = map_elites.update(rep, est, subkey)
        return (rep, est, key), metrics
        
    # QDAX Warmup
    print("JIT Compiling QDAX...")
    carry = (repertoire, emitter_state, k4)
    carry_warmup, _ = jax.lax.scan(step, carry, None, length=2)
    jax.block_until_ready(carry_warmup)
    
    # QDAX Run
    print("Running QDAX Benchmark...")
    t0 = time.time()
    carry, metrics = jax.lax.scan(step, carry, None, length=NUM_GENS)
    jax.block_until_ready(carry)
    print(f"QDAX Done in {time.time() - t0:.3f}s")
    
    rep = carry[0]
    best_fitness = float(jnp.max(rep.fitnesses))
    qd_score = float(jnp.sum(jnp.where(rep.fitnesses > -jnp.inf, rep.fitnesses, 0.0)))
    coverage = float(jnp.sum(rep.fitnesses > -jnp.inf))
    return coverage, best_fitness, qd_score

def run_malthusjax(k):
    print("\n--- Running MalthusJAX Benchmark ---")
    k1, k2, k3, k4 = jax.random.split(k, 4)
    
    genome_config = RealGenomeConfig(shape=(NUM_VARS,), bounds=(-5.12, 5.12))
    
    mutation = GaussianMutation(
        mutation_strength=1.0,
        mutation_rate=0.5
    )
    crossover = UniformCrossover(
        crossover_rate=1.0
    )
    
    mut_emitter = GeneticMutationEmitter(_batch_size=BATCH_SIZE//2, mutation=mutation, genome_config=genome_config)
    cx_emitter = GeneticCrossoverEmitter(_batch_size=BATCH_SIZE//2, crossover=crossover, genome_config=genome_config)
    emitter = MixingEmitter(emitter_a=mut_emitter, emitter_b=cx_emitter)
    
    evaluator = MJX_Evaluator(config=None, data=None)
    
    engine = MapElitesEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=MapElitesEngineParams(pop_size=BATCH_SIZE, num_generations=NUM_GENS)
    )
    
    centroids = compute_cvt_centroids(2, 5000, NUM_CENTROIDS, 0.0, 1.0, k1)
    init_pop = genome_config.init_population(k2, BATCH_SIZE)
    engine_state = engine.init_state(k3, init_pop, centroids=centroids)
    
    # Scan loop
    @jax.jit
    def step_mjx(carry, unused):
        state = carry
        new_state, history = engine.step(state)
        return new_state, history
        
    print("JIT Compiling MalthusJAX...")
    carry_warmup, _ = jax.lax.scan(step_mjx, engine_state, None, length=2)
    jax.block_until_ready(carry_warmup)
    
    print("Running MalthusJAX Benchmark...")
    t0 = time.time()
    final_state, metrics = jax.lax.scan(step_mjx, engine_state, None, length=NUM_GENS)
    jax.block_until_ready(final_state)
    print(f"MalthusJAX Done in {time.time() - t0:.3f}s")
    
    rep = final_state.repertoire
    best_fitness = float(jnp.max(rep.fitnesses))
    qd_score = float(jnp.sum(jnp.where(rep.fitnesses > -jnp.inf, rep.fitnesses, 0.0)))
    coverage = float(jnp.sum(rep.fitnesses > -jnp.inf))
    return coverage, best_fitness, qd_score

def main():
    k = jax.random.PRNGKey(42)
    qdax_cov, qdax_fit, qdax_qd = run_qdax(k)
    mjx_cov, mjx_fit, mjx_qd = run_malthusjax(k)
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"QDAX       -> Coverage: {qdax_cov:g}, Best Fitness: {qdax_fit:.2f}, QD Score: {qdax_qd:.2f}")
    print(f"MalthusJAX -> Coverage: {mjx_cov:g}, Best Fitness: {mjx_fit:.2f}, QD Score: {mjx_qd:.2f}")

if __name__ == "__main__":
    main()
