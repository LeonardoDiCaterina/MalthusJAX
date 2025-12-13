import time
import jax
import jax.numpy as jnp
import jax.random as jar
import pandas as pd
from evosax.algorithms.population_based.simple_ga import SimpleGA
from evosax.problems import BBOBProblem

# --- MalthusJAX Imports ---
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation

# ==========================================
# CONFIGURATION (Matches Notebook)
# ==========================================
PROBLEM_NAME = "sphere"
DIMENSIONS = 10
BOUNDS = (-5.0, 5.0)
FIXED_PROBLEM_SEED = 0

POP_SIZE = 100
NUM_GENS = 1000  # Increased for better timing resolution
CROSSOVER_RATE = 0.5
ELITE_RATIO = 0.2
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.1
ELITE_POOL_SIZE = int(POP_SIZE * ELITE_RATIO)

def run_malthusjax_benchmark():
    print(f"\n[MalthusJAX] Setting up Engine...")
    
    # 1. Configure Components
    genome_config = RealGenomeConfig(length=DIMENSIONS, bounds=BOUNDS)
    
    # BBOB Config (Maximization=True to flip sign, as per notebook)
    eval_config = BBOBConfig(
        fn_name=PROBLEM_NAME,
        num_dims=DIMENSIONS,
        seed=FIXED_PROBLEM_SEED,
        maximize=True 
    )
    evaluator = BBOBEvaluator.create(eval_config)
    
    selection = ElitePoolSelection(
        num_selections=POP_SIZE, 
        elite_k=ELITE_POOL_SIZE, 
        input_length=POP_SIZE
    )
    crossover = UniformCrossover(num_offspring=2, crossover_rate=CROSSOVER_RATE)
    mutation = GaussianMutation(
        num_offspring=1, 
        mutation_rate=MUTATION_RATE, 
        mutation_strength=MUTATION_STRENGTH
    )
    
    engine_params = GeneticEngineParams(
        pop_size=POP_SIZE,
        num_generations=NUM_GENS,
        elitism=0 # Elitism handled by selection/merge in this config
    )
    
    engine = GeneticEngine(
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        genome_config=genome_config,
        engine_params=engine_params,
    )
    
    # 2. Init State
    key = jar.PRNGKey(42)
    state = engine.init_state(key)
    
    # 3. Warmup (Compilation)
    print("[MalthusJAX] Compiling...", end="", flush=True)
    t0 = time.time()
    _final, _, _ = engine.run(state, compile=True)
    _ = _final.best_fitness.block_until_ready()
    compile_time = time.time() - t0
    print(f" Done ({compile_time:.4f}s)")
    
    # 4. Execution (Hot Run)
    # Re-init state to ensure clean run
    state = engine.init_state(key)
    
    t0 = time.time()
    final_state, _, _ = engine.run(state, compile=True)
    _ = final_state.best_fitness.block_until_ready()
    exec_time = time.time() - t0
    
    throughput = NUM_GENS / exec_time
    # Negate fitness back because MalthusJAX maximized the negative cost
    final_fitness = -final_state.best_fitness 
    
    return {
        "Framework": "MalthusJAX",
        "Compile Time (s)": compile_time,
        "Execution Time (s)": exec_time,
        "Generations/Sec": throughput,
        "Final Fitness (Cost)": float(final_fitness)
    }

def run_evosax_benchmark():
    print(f"\n[Evosax] Setting up Strategy...")
    
    # 1. Setup Problem & Strategy
    # Evosax native BBOB
    problem = BBOBProblem(PROBLEM_NAME, num_dims=DIMENSIONS, seed=FIXED_PROBLEM_SEED)
    
    strategy = SimpleGA(
        population_size=POP_SIZE,
        solution=problem.sample(jar.PRNGKey(0))
    )
    es_params = strategy.default_params.replace(crossover_rate=CROSSOVER_RATE)
    
    # 2. Define the Scan Loop (Equivalent to MalthusJAX engine.run)
    def step_impl(carry, _):
        evosax_state, param_state, rng = carry
        rng_ask, rng_eval, rng_tell, rng_next = jax.random.split(rng, 4)
        
        # Ask
        x, evosax_state = strategy.ask(rng_ask, evosax_state, es_params)
        # Eval
        fitness, new_param_state, _ = problem.eval(rng_eval, x, param_state)
        # Tell
        evosax_state, metrics = strategy.tell(rng_tell, x, fitness, evosax_state, es_params)
        
        return (evosax_state, new_param_state, rng_next), evosax_state.best_fitness
    
    # 3. Initialization
    rng = jar.PRNGKey(42)
    rng, rng_init, rng_pop = jax.random.split(rng, 3)
    
    # Custom init population to match MalthusJAX bounds logic if possible, 
    # but strategy.init does its own thing.
    initial_pop = jax.random.uniform(rng_pop, (POP_SIZE, DIMENSIONS), minval=BOUNDS[0], maxval=BOUNDS[1])
    initial_fitness = jnp.full((POP_SIZE,), jnp.inf)
    
    evosax_state = strategy.init(rng_init, initial_pop, initial_fitness, es_params)
    prob_state = problem.init(jar.PRNGKey(FIXED_PROBLEM_SEED))
    
    carry_init = (evosax_state, prob_state, rng)
    
    # 4. Compilation (Warmup)
    print("[Evosax] Compiling...", end="", flush=True)
    scan_fn = jax.jit(lambda c: jax.lax.scan(step_impl, c, None, length=NUM_GENS))
    
    t0 = time.time()
    final_carry, _ = scan_fn(carry_init)
    # Block on a specific scalar to force sync
    _ = final_carry[0].best_fitness.block_until_ready()
    compile_time = time.time() - t0
    print(f" Done ({compile_time:.4f}s)")
    
    # 5. Execution (Hot Run)
    t0 = time.time()
    final_carry, _ = scan_fn(carry_init)
    _ = final_carry[0].best_fitness.block_until_ready()
    exec_time = time.time() - t0
    
    throughput = NUM_GENS / exec_time
    final_fitness = final_carry[0].best_fitness
    
    return {
        "Framework": "Evosax",
        "Compile Time (s)": compile_time,
        "Execution Time (s)": exec_time,
        "Generations/Sec": throughput,
        "Final Fitness (Cost)": float(final_fitness)
    }

def main():
    print("==================================================")
    print("   MALTHUSJAX vs EVOSAX PERFORMANCE BENCHMARK")
    print("==================================================")
    print(f"Generations: {NUM_GENS} | Pop Size: {POP_SIZE} | Dims: {DIMENSIONS}")
    
    # Run Benchmarks
    res_mjx = run_malthusjax_benchmark()
    res_es = run_evosax_benchmark()
    
    # Display Results
    df = pd.DataFrame([res_mjx, res_es])
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False, float_format="%.4f"))
    print("="*60)
    
    # Validation
    fit_diff = abs(res_mjx['Final Fitness (Cost)'] - res_es['Final Fitness (Cost)'])
    print(f"\nFitness Difference: {fit_diff:.4f}")
    if fit_diff < 5.0:
        print("✅ Algorithmic Equivalence: PASS (Converged to similar optima)")
    else:
        print("⚠️ Algorithmic Equivalence: DIVERGENCE (Check operator params)")
        
    speed_factor = res_mjx['Generations/Sec'] / res_es['Generations/Sec']
    print(f"Speedup Factor (MalthusJAX / Evosax): {speed_factor:.2f}x")

if __name__ == "__main__":
    main()