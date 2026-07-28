import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorneat", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "QDAX")))

import jax
import jax.numpy as jnp
from tensorneat.genome import DefaultGenome, DefaultNode, DefaultConn
from tensorneat.common import State

from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.core.fitness.tensorneat import TensorNeatQDEvaluator
from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.tensorneat_variants import TensorNeatMutationEmitter, TensorNeatCrossoverEmitter
from malthusjax.core.genome.tensorneat_genome import TensorNeatPopulation, TensorNeatGenome
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

def main():
    print("Initializing TensorNEAT Components...")
    tn_genome = DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn()
    )

    print("Building MalthusJAX Compositional Emitter...")
    # Half batch mutation, half batch crossover
    mut_emitter = TensorNeatMutationEmitter(_batch_size=8, genome=tn_genome)
    cx_emitter = TensorNeatCrossoverEmitter(_batch_size=8, genome=tn_genome)
    emitter = MixingEmitter(emitter_a=mut_emitter, emitter_b=cx_emitter)
    
    # 3. Setup Dummy Objective Function
    def objective_fn(nodes, conns):
        batch_size = nodes.shape[0]
        fitnesses = jnp.sum(~jnp.isnan(nodes[:, :, 0]), axis=1).astype(jnp.float32)
        desc_x = jnp.clip(fitnesses / 10.0, 0.0, 1.0)
        desc_y = jnp.clip((fitnesses ** 2) / 100.0, 0.0, 1.0)
        descriptors = jnp.stack([desc_x, desc_y], axis=-1)
        return fitnesses, descriptors
        
    evaluator = TensorNeatQDEvaluator(objective_function=objective_fn, config=None, data=None)
    
    engine_params = MapElitesEngineParams(
        pop_size=16,
        num_generations=100
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
        num_init_cvt_samples=1000, 
        num_centroids=50, 
        minval=0.0, 
        maxval=1.0, 
        key=k1
    )
    
    state = State(randkey=k1)
    state = tn_genome.setup(state)
    
    init_keys = jax.random.split(k2, engine_params.pop_size)
    init_nodes, init_conns = jax.vmap(tn_genome.initialize, in_axes=(None, 0))(state, init_keys)
    
    genes = TensorNeatGenome(values=(init_nodes, init_conns))
    initial_pop = TensorNeatPopulation(
        genes=genes,
        fitness=jnp.full(engine_params.pop_size, -jnp.inf),
        config=None
    )
    
    print("Initializing MAP-Elites Engine...")
    import time
    t0 = time.time()
    engine_state = engine.init_state(k3, initial_pop, centroids=centroids)
    print(f"Calling engine.init_state... Done in {time.time()-t0:.3f}s")
    
    print("Running Compositional QD Evolution for 100 generations...")
    final_state, history, elapsed = engine.run(engine_state, time_it=True, compile=True, verbose=True)
    
    print(f"\nEvolution Complete in {elapsed:.2f}s!")
    print(f"Final Best Fitness: {float(final_state.best_fitness):.1f}")
    print(f"Final QD Score: {history.qd_score[-1]}")
    print(f"Final Map Coverage: {history.coverage[-1] * 100:.2f}% ({jnp.sum(final_state.repertoire.fitnesses > -jnp.inf)} cells)")

if __name__ == "__main__":
    main()
