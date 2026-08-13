import time

import jax
import jax.numpy as jnp

# Check if QDAX is installed
try:
    import qdax
    from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
    from qdax.core.emitters.mutation_operators import isoline_variation
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
except ImportError:
    print("QDAX is not installed. Run `pip install qdax` first.")
    exit(1)

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter


# ─── QD Dummy Evaluator ───────────────────────────────────────────────────────
class QDDummyEvaluator(BaseQDEvaluator):
    def evaluate(self, genome):
        # Unused for population eval
        pass

    def evaluate_population(self, pop: BasePopulation) -> BasePopulation:
        x = pop.genes.values
        fitness = -jnp.sum(x**2, axis=-1)  # Maximize negative sphere
        descriptors = x[:, :2]  # First two dims
        info = {"descriptors": descriptors}
        return pop.replace(fitness=fitness, info=info)


def scoring_fn(x, random_key):
    # Match the DummyEvaluator output exactly
    fitness = -jnp.sum(x**2, axis=-1)
    descriptors = x[:, :2]
    return fitness, descriptors, {}


import argparse

# ─── Setup Parameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 128
NUM_DIM = 9
NUM_GENS = 500
GRID_SHAPE = (10, 10)
MIN_PARAM = -5.0
MAX_PARAM = 5.0

key = jax.random.PRNGKey(42)

# Compute centroids
centroids = compute_euclidean_centroids(
    grid_shape=GRID_SHAPE,
    minval=MIN_PARAM,
    maxval=MAX_PARAM,
)


# ─── 1. Bench QDAX (Reference) ────────────────────────────────────────────────
def bench_qdax(runs=5):
    emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=lambda x, y, z: isoline_variation(x, y, z, iso_sigma=0.1, line_sigma=0.2),
        variation_percentage=1.0,
        batch_size=BATCH_SIZE,
    )

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=lambda r: jnp.max(r.fitnesses),
    )

    init_pop = jnp.zeros((BATCH_SIZE, NUM_DIM))
    repertoire, emitter_state, _ = map_elites.init(init_pop, centroids, key)

    @jax.jit
    def run_fn(r, es, rk):
        def scan_step(carry, _):
            c_r, c_es, c_k = carry
            c_k, subk = jax.random.split(c_k)
            n_r, n_es, _ = map_elites.update(c_r, c_es, subk)
            return (n_r, n_es, c_k), ()

        (final_r, _, _), _ = jax.lax.scan(scan_step, (r, es, rk), jnp.arange(NUM_GENS))
        return final_r

    out = run_fn(repertoire, emitter_state, key)
    out.fitnesses.block_until_ready()

    start = time.perf_counter()
    for i in range(runs):
        out = run_fn(repertoire, emitter_state, jax.random.PRNGKey(i))
        out.fitnesses.block_until_ready()
    end = time.perf_counter()

    ms = (end - start) / runs * 1000
    print(f"  0. QDAX Reference MAP-Elites:          {ms:8.2f} ms")
    return ms


# ─── 2. Bench MalthusJAX ──────────────────────────────────────────────────────
def bench_malthusjax(runs=5):
    evaluator = QDDummyEvaluator(config=None, data=None)
    genome_config = RealGenomeConfig(bounds=(MIN_PARAM, MAX_PARAM))

    # Exact same emitter
    replica_emitter = QDAXReplicaMixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=lambda x, y, z: isoline_variation(x, y, z, iso_sigma=0.1, line_sigma=0.2),
        variation_percentage=1.0,
        _batch_size=BATCH_SIZE,
        genome_config=genome_config,
    )

    engine_params = MapElitesEngineParams(
        pop_size=BATCH_SIZE, num_generations=NUM_GENS, key_derivation="qdax_replica", maximize=True
    )

    engine = MapElitesEngine(
        emitter=replica_emitter, evaluator=evaluator, engine_params=engine_params
    )

    init_genes = RealGenome(values=jnp.zeros((BATCH_SIZE, NUM_DIM)))
    init_pop = RealPopulation(genes=init_genes, fitness=jnp.zeros(BATCH_SIZE), config=genome_config)

    state = engine.init_state(key, init_pop, centroids)

    @jax.jit
    def run_fn(rng):
        def scan_step(carry, _):
            st = carry
            new_st, _ = engine.step(st)
            return new_st, ()

        final_carry, _ = jax.lax.scan(scan_step, state.replace(rng_key=rng), jnp.arange(NUM_GENS))
        return final_carry

    out = run_fn(key)
    out.best_fitness.block_until_ready()

    start = time.perf_counter()
    for i in range(runs):
        out = run_fn(jax.random.PRNGKey(i))
        out.best_fitness.block_until_ready()
    end = time.perf_counter()

    ms = (end - start) / runs * 1000
    print(f"  1. MalthusJAX MapElitesEngine:         {ms:8.2f} ms")
    return ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()

    runs = 5
    if args.smoke:
        NUM_GENS = 10
        runs = 2

    print("=====================================================")
    print("  QD ARCHITECTURE ABLATION: QDAX vs MalthusJAX")
    print(f"  D={NUM_DIM}, Pop={BATCH_SIZE}, Gens={NUM_GENS}, Grid={GRID_SHAPE}")
    print("=====================================================")

    bench_qdax(runs=runs)
    bench_malthusjax(runs=runs)

    print("=====================================================")
