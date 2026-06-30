import jax
import jax.numpy as jnp
import sys

# Add local Kozax path
sys.path.append('/Users/leonardodicaterina/.gemini/antigravity-ide/brain/b61d8dbc-c0e6-4408-9466-2d7ebff50ada/scratch/Kozax')

from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction
from malthusjax.composer.kozax_adapter import build_kozax_engine
from malthusjax.composer.adapters import EvalMode

class SimpleSumFitness(BaseFitnessFunction):
    def __call__(self, candidate: str, data: tuple, tree_evaluator) -> jnp.ndarray:
        # data contains the inputs (X) and targets (Y)
        X, Y = data
        jax.debug.print("X shape: {x}, Y shape: {y}", x=X.shape, y=Y.shape)
        # Evaluate the symbolic tree on X
        pred = jax.vmap(lambda c, x: tree_evaluator(c, x), in_axes=(None, 0))(candidate, X)
        # We want to minimize mean squared error
        return jnp.mean((pred.flatten() - Y)**2)

if __name__ == "__main__":
    # Create simple regression data: f(x) = x_0 + x_1
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, (100, 2))
    Y = X[:, 0] + X[:, 1]
    data = (X, Y)
    
    # Define GP operators
    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
    ]
    variable_list = [["x0", "x1"]]
    
    # Setup Kozax GP
    num_generations = 5
    population_size = 20
    num_populations = 2
    
    fitness_function = SimpleSumFitness()
    
    gp = GeneticProgramming(
        num_generations, 
        population_size, 
        fitness_function, 
        operator_list, 
        variable_list, 
        jnp.array([1]), # layer_sizes
        num_populations=num_populations,
    )
    
    engine = build_kozax_engine(
        strategy_obj=gp,
        evaluator=data,
        generations=num_generations,
        eval_mode=EvalMode.NATIVE,
        maximize=False
    )
    
    print("Running Kozax through MalthusJAX...")
    results = engine.run_once(jax.random.PRNGKey(42))
    
    print("Optimization finished!")
    print(f"Best fitness (MSE): {results['summary']['best_fitness']}")
    print(f"Total Evaluated: {results['summary']['total_evaluations']}")
    for gen in results['history']:
        print(f"Generation {gen['generation']}: Best: {gen['best_fitness']:.4f} | Mean: {gen['mean_fitness']:.4f}")
