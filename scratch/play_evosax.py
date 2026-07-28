import jax
import jax.numpy as jnp
from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig

def main():
    print("Testing CMA_ES from Evosax...")
    # BBOB evaluator for a simple 5D Rastrigin problem
    config = BBOBConfig(fn_name="Rastrigin", num_dims=5, maximize=False)
    evaluator = BBOBEvaluator.create(config)

    # Build adapter
    engine = build_evosax_engine(
        strategy_name="CMA_ES",
        evaluator=evaluator,
        pop_size=50,
        generations=10,
        bounds=(-5.0, 5.0),
        maximize=False,
    )
    
    # Run the engine
    key = jax.random.PRNGKey(42)
    results = engine.run_once(key)
    
    print("Run completed successfully!")
    print(f"Final best fitness: {results['summary']['best_fitness']}")

if __name__ == "__main__":
    main()
