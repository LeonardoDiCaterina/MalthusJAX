import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorneat", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "QDAX")))

import jax
import jax.numpy as jnp
from tensorneat.genome import DefaultGenome, DefaultNode, DefaultConn
from tensorneat.genome.operations.mutation import DefaultMutation
from tensorneat.genome.operations.crossover import DefaultCrossover
from tensorneat.common import State

from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter
from malthusjax.core.fitness.tensorneat import TensorNeatQDEvaluator
from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

def main():
    print("Initializing TensorNEAT Components...")
    # 1. Setup TensorNEAT Core
    tn_genome = DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn()
    )
    mutation = DefaultMutation()
    crossover = DefaultCrossover()
    
    # 2. Setup MalthusJAX QD Emitter
    print("Building MalthusJAX QD Emitter...")
    emitter = TensorNeatEmitter(
        _batch_size=16,
        genome=tn_genome,
        mutation=mutation,
        crossover=crossover
    )
    
    # 3. Setup Dummy Objective Function
    def objective_fn(nodes, conns):
        batch_size = nodes.shape[0]
        # Dummy fitness: Just number of active nodes
        fitnesses = jnp.sum(~jnp.isnan(nodes[:, :, 0]), axis=1).astype(jnp.float32)
        # Dummy descriptors: random 2D points between 0 and 1
        # Use pseudo-random based on fitness to make it deterministic but varied
        desc_x = jnp.clip(fitnesses / 10.0, 0.0, 1.0)
        desc_y = jnp.clip((fitnesses ** 2) / 100.0, 0.0, 1.0)
        descriptors = jnp.stack([desc_x, desc_y], axis=-1)
        return fitnesses, descriptors
        
    evaluator = TensorNeatQDEvaluator(objective_function=objective_fn, config=None, data=None)
    
    # 4. Compute CVT Centroids for QDAX Map
    print("Computing QDAX CVT Centroids...")
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=50,
        minval=0.0,
        maxval=1.0,
        key=jax.random.PRNGKey(42)
    )
    
    # 5. Initialize Engine
    print("Initializing MAP-Elites Engine...")
    params = MapElitesEngineParams(
        pop_size=16,  # batch size
        num_generations=100
    )
    
    engine = MapElitesEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=params
    )
    
    # 6. Initialize Population manually using TensorNEAT
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    state = State(randkey=k1)
    state = tn_genome.setup(state)
    
    init_keys = jax.random.split(k2, params.pop_size)
    init_nodes, init_conns = jax.vmap(tn_genome.initialize, in_axes=(None, 0))(state, init_keys)
    
    initial_pop = TensorNeatPopulation(
        genes=TensorNeatGenome(values=(init_nodes, init_conns)),
        fitness=jnp.full(params.pop_size, -jnp.inf),
        config=None
    )
    
    print("Calling engine.init_state...")
    # Initialize Engine State
    engine_state = engine.init_state(k3, initial_pop, centroids)
    
    print("Initial Best Fitness:", engine_state.best_fitness)
    print("Initial Repertoire Size:", jnp.sum(engine_state.repertoire.fitnesses > -jnp.inf))
    
    # 7. Run Evolution!
    print(f"Running Native QD Evolution for {params.num_generations} generations...")
    final_state, history, elapsed = engine.run(engine_state, time_it=True, compile=True, verbose=True)
    
    print(f"\nEvolution Complete in {elapsed:.2f}s!")
    print(f"Final Best Fitness: {final_state.best_fitness}")
    print(f"Final QD Score: {history.qd_score[-1]}")
    print(f"Final Map Coverage: {history.coverage[-1] * 100:.2f}% ({jnp.sum(final_state.repertoire.fitnesses > -jnp.inf)} cells)")
    
if __name__ == "__main__":
    main()
