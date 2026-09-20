"""
Showcase 1: Framework Dissection (Adapter -> Composable Evaluator -> Native Problem)

This script demonstrates MalthusJAX's architectural flexibility. 
We take a third-party framework's strategy (TensorNEAT's NEAT algorithm), wrap it in our 
Universal Engine Adapter (`build_tensorneat_engine`), and evaluate it using our composable 
`TensorNeatEvaluator` — which internally wraps the original TensorNEAT problem.

We have effectively "broken apart" the framework and rebuilt it, routing execution 
through MalthusJAX's standard evaluation protocol (EvalMode.MALTHUSJAX).
"""

import jax
import jax.numpy as jnp
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import XOR

from malthusjax.composer.tensorneat_adapter import build_tensorneat_engine
from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.environments import TensorNEATProblemWrapper
from malthusjax.core.fitness.composable.evaluators import TensorNeatEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from plugins.tensor_neat_transform import TensorNeatTransform


from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter


def main():
    print("=== MalthusJAX Framework Dissection Showcase ===")
    print("1. Constructing the Composable TensorNeatEvaluator...")
    
    pop_size = 100
    problem = XOR()
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)
    algorithm = NEAT(pop_size=pop_size, species_size=2, genome=genome)

    # 1. Evaluator (MalthusJAX composable wrapper)
    evaluator = TensorNeatEvaluator(
        env=TensorNEATProblemWrapper(problem=problem),
        transform=TensorNeatTransform(algorithm=algorithm),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
        forward_fn=algorithm.forward,
        seed=42,
    )

    print("2. Constructing the Native TensorNeatEmitter...")
    # 2. Emitter (MalthusJAX native operator)
    emitter = TensorNeatEmitter(_batch_size=pop_size, genome=genome)

    print("3. Executing the Rebuilt Evolution Loop in Pure JAX...")
    
    # 3. Initialize state
    from tensorneat.common import State
    key = jax.random.PRNGKey(42)
    tn_state = State(randkey=key)
    tn_state = algorithm.setup(tn_state)
    pop_values = algorithm.ask(tn_state)
    
    initial_pop = TensorNeatPopulation(
        genes=TensorNeatGenome(values=pop_values),
        fitness=jnp.full(pop_size, -jnp.inf),
        config=None,
        info={}
    )

    # Evaluate initial population
    evaluated_pop = evaluator.evaluate_population(initial_pop)
    
    # Create Dummy Repertoire using QDAX's container
    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
    dummy_descriptors = jnp.zeros((pop_size, 1))
    repertoire = MapElitesRepertoire.init(
        genotypes=getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
        fitnesses=evaluated_pop.fitness,
        descriptors=dummy_descriptors,
        centroids=jnp.array([[0.0]]),
    )

    emitter_state = emitter.init(key, evaluated_pop, params=None)
    best_fitness = jnp.max(evaluated_pop.fitness)

    print(f"Generation 0 - Best Fitness: {best_fitness:.4f}")

    # 4. Standard scan loop using purely MalthusJAX primitives
    @jax.jit
    def step(carry, _):
        key, em_state, rep, best_fit = carry
        key, ask_key_seed = jax.random.split(key)
        ask_key = jax.random.split(ask_key_seed, emitter.num_keys())
        
        # ASK (via MalthusJAX Emitter)
        offspring, new_em_state = emitter.ask(em_state, rep, ask_key)
        
        # EVALUATE (via MalthusJAX Composable Evaluator)
        evaluated = evaluator.evaluate_population(offspring)
        
        # TELL (via MalthusJAX Emitter)
        new_em_state = emitter.tell(
            new_em_state,
            rep,
            evaluated,
            evaluated.fitness,
            dummy_descriptors,
            key
        )

        # Update repertoire
        new_rep = rep.add(
            getattr(evaluated.genes, "values", evaluated.genes),
            jnp.zeros((pop_size, 1)),
            evaluated.fitness,
        )
        
        new_best = jnp.maximum(best_fit, jnp.max(evaluated.fitness))
        return (key, new_em_state, new_rep, new_best), new_best

    carry = (key, emitter_state, repertoire, best_fitness)
    for gen in range(1, 11):
        import numpy as np
        # Need to ensure JAX tracers don't corrupt the Python-loop PRNG
        carry, step_best = step(carry, None)
        if gen % 2 == 0:
            print(f"Generation {gen} - Best Fitness: {step_best:.4f}")

    print("\nSuccess! We evolved a TensorNEAT genome using the MalthusJAX Emitter,")
    print("and evaluated it exclusively through MalthusJAX's composable Evaluator layer.")

if __name__ == "__main__":
    main()
