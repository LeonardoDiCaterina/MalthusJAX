"""Benchmark: Pure TensorNEAT (Adapter) vs MalthusJAX Native (TensorNeatEmitter)."""

import time

import jax
import jax.numpy as jnp

try:
    import tensorneat

    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False


def main():
    if not TENSORNEAT_AVAILABLE:
        print("TensorNEAT is not installed. Skipping benchmark.")
        return

    from malthusjax.composer import Composer

    # Configuration for both pipelines
    pop_size = 128
    generations = 10

    # -------------------------------------------------------------------------
    # Pipeline 1: Pure TensorNEAT Adapter
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" PIPELINE 1: TensorNEAT Pure (Adapter)")
    print("=" * 50)

    composer = Composer.create_default()
    result = composer.quick_run(
        backend="tensorneat",
        fitness="tensorneat:problem=xor",
        strategy=f"tensorneat:algorithm=neat:pop_size={pop_size}:species_size=2:max_nodes=10:max_conns=20",
        eval_mode="native",
        generations=generations,
    )

    run = result.runs[0]
    adapter_time = run.duration_seconds
    metrics = run.metrics
    print(f"Adapter Final Max Fitness: {metrics['best_fitness']:.4f}")
    print(f"Adapter Execution Time (incl compile): {adapter_time:.4f} seconds")

    # -------------------------------------------------------------------------
    # Pipeline 2: MalthusJAX Native
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" PIPELINE 2: MalthusJAX Native (TensorNeatEmitter)")
    print("=" * 50)

    from tensorneat.algorithm import NEAT
    from tensorneat.genome import DefaultGenome
    from tensorneat.problem import XOR

    from malthusjax.core.fitness.qd.tensorneat_evaluator import (
        TensorNeatEvaluatorConfig,
        TensorNeatQDEvaluator,
    )
    from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter

    # Setup the native TensorNEAT operators for the evaluator and emitter
    problem = XOR()
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)
    algorithm = NEAT(pop_size=pop_size, species_size=2, genome=genome)

    evaluator = TensorNeatQDEvaluator.create(
        algorithm=algorithm,
        problem=problem,
        forward_fn=algorithm.forward,
        config=TensorNeatEvaluatorConfig(seed=42),
    )

    # The MapElites Repertoire will store the TensorNEAT dynamic graphs!
    emitter = TensorNeatEmitter(_batch_size=pop_size, genome=genome)

    # To run the Native Engine, we need an initial population of graphs.
    from tensorneat.common import State

    from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation

    key = jax.random.PRNGKey(42)
    tn_state = State(randkey=key)
    tn_state = algorithm.setup(tn_state)
    pop_values = algorithm.ask(tn_state)

    initial_genes = TensorNeatGenome(values=pop_values)
    initial_pop = TensorNeatPopulation(
        genes=initial_genes, fitness=jnp.full(pop_size, -jnp.inf), config=None, info={}
    )

    # MapElites initialization
    start_time = time.time()

    # Make sure qdax is imported for MapElitesRepertoire
    try:
        from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
    except ImportError:
        MapElitesRepertoire = Any

    @jax.jit
    def init_step(k):
        # We evaluate the initial population to get fitnesses
        evaluated_pop = evaluator.evaluate_population(initial_pop)

        # Inject Dummy descriptors (e.g. ones) since XOR doesn't return them
        new_info = dict(evaluated_pop.info)
        new_info["descriptors"] = jnp.full((pop_size, 1), 0.5)
        evaluated_pop = evaluated_pop.replace(info=new_info)

        # Initialize Repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
            fitnesses=evaluated_pop.fitness,
            descriptors=evaluated_pop.info["descriptors"],
            centroids=jnp.array([[0.5]]),
        )
        # Initialize Emitter state
        emitter_state = emitter.init(k, evaluated_pop, params=None)

        best_genome_idx = jnp.argmax(evaluated_pop.fitness)
        best_genome_values = jax.tree_util.tree_map(
            lambda x: x[best_genome_idx],
            getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
        )
        if hasattr(evaluated_pop.genes, "replace"):
            best_genome = evaluated_pop.genes.replace(values=best_genome_values)
        else:
            best_genome = best_genome_values

        from malthusjax.engine.qd.map_elites import MapElitesState

        return MapElitesState(
            population=evaluated_pop,
            best_genome=best_genome,
            generation=0,
            best_fitness=jnp.max(evaluated_pop.fitness),
            repertoire=repertoire,
            emitter_state=emitter_state,
            rng_key=k,
        )

    @jax.jit
    def train_step(state):
        # We manually run a MapElites step, injecting dummy descriptors after evaluation
        k, key_ask_seed, key_eval = jax.random.split(state.rng_key, 3)
        key_ask = jax.random.split(key_ask_seed, emitter.num_keys())

        # Ask
        offspring_pop, new_emitter_state = emitter.ask(
            state.emitter_state, state.repertoire, key_ask
        )

        # Evaluate
        evaluated_pop = evaluator.evaluate_population(offspring_pop)
        new_info = dict(evaluated_pop.info)
        new_info["descriptors"] = jnp.full((pop_size, 1), 0.5)
        evaluated_pop = evaluated_pop.replace(info=new_info)

        # Tell
        new_repertoire = state.repertoire.add(
            getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
            evaluated_pop.info["descriptors"],
            evaluated_pop.fitness,
        )

        best_genome_idx = jnp.argmax(evaluated_pop.fitness)
        best_genome_values = jax.tree_util.tree_map(
            lambda x: x[best_genome_idx],
            getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
        )
        if hasattr(evaluated_pop.genes, "replace"):
            best_genome = evaluated_pop.genes.replace(values=best_genome_values)
        else:
            best_genome = best_genome_values

        return state.replace(
            population=evaluated_pop,
            best_genome=best_genome,
            generation=state.generation + 1,
            best_fitness=jnp.maximum(state.best_fitness, jnp.max(evaluated_pop.fitness)),
            repertoire=new_repertoire,
            emitter_state=new_emitter_state,
            rng_key=k,
        )

    state = init_step(key)

    import tqdm

    for _ in tqdm.trange(generations, desc="Native Step"):
        state = train_step(state)

    native_time = time.time() - start_time
    print(f"Native Final Max Fitness: {state.best_fitness:.4f}")
    print(f"Native Execution Time: {native_time:.4f} seconds")


if __name__ == "__main__":
    main()
