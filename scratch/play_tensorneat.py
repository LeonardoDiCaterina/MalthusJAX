import jax
import jax.numpy as jnp
from tensorneat.algorithm.neat import NEAT
from tensorneat.problem import BaseProblem
from malthusjax.composer.tensorneat_adapter import build_tensorneat_engine
from malthusjax.composer.adapters import EvalMode

class SimpleSumProblem(BaseProblem):
    
    def setup(self, state=None):
        return state or {}

    def evaluate(self, state, randkey, forward_fn, params):
        # forward_fn is the network's forward function
        # params are the network weights
        # We just test the network with a random input
        inputs = jax.random.normal(randkey, (2,))
        outputs = forward_fn(state, params, inputs)
        # Our "fitness" is how close the output is to 0
        loss = jnp.mean(jnp.square(outputs))
        return -loss # we want to maximize fitness (minimize loss)

if __name__ == "__main__":
    from tensorneat.genome import DefaultGenome
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=100, max_conns=200)
    algorithm = NEAT(
        genome=genome,
        pop_size=50, 
        species_size=5,
    )
    problem = SimpleSumProblem()
    
    engine = build_tensorneat_engine(
        algorithm=algorithm,
        evaluator=problem,
        generations=10,
        eval_mode=EvalMode.NATIVE,
        maximize=True
    )
    
    print("Running TensorNEAT through MalthusJAX...")
    results = engine.run_once(jax.random.PRNGKey(42))
    
    print("Optimization finished!")
    print(f"Best fitness: {results['summary']['best_fitness']}")
    print(f"Total Evaluated: {results['summary']['total_evaluations']}")
    for gen in results['history']:
        print(f"Generation {gen['generation']}: Best: {gen['best_fitness']:.4f} | Mean: {gen['mean_fitness']:.4f}")
