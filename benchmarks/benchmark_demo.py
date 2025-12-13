import time
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jar
import matplotlib.pyplot as plt
import seaborn as sns

# --- Evosax Imports ---
from evosax.algorithms.population_based.simple_ga import SimpleGA
from evosax.problems import BBOBProblem

# --- MalthusJAX Imports ---
# FIX 1: Import RealGenome class explicitly
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation

# ==========================================
# CONFIGURATION
# ==========================================
POP_SIZE = 2048
NUM_SEEDS = 30
NUM_GENS_FAST = 10     # For Python loops (Speed test only)
NUM_GENS_FULL = 10   # For Compiled loops (Convergence test)
DIMENSIONS = 20
FIXED_PROBLEM_SEED = 0 # Crucial: Both frameworks must solve the same landscape

CROSSOVER_RATE = 0.5
ELITE_RATIO = 0.2
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.1
ELITE_POOL_SIZE = int(POP_SIZE * ELITE_RATIO)

# ==========================================
# SETUP HELPERS
# ==========================================
def setup_malthus(seed):
    """Creates MalthusJAX components."""
    genome_config = RealGenomeConfig(length=DIMENSIONS, bounds=(-5.0, 5.0))
    
    # SETUP EVALUATOR (Correctly)
    # 1. Create config
    # We ensure maximize=True is passed clearly
    eval_config = BBOBConfig(
        fn_name="sphere", 
        num_dims=DIMENSIONS, 
        seed=FIXED_PROBLEM_SEED, 
        maximize=True 
    )
    evaluator = BBOBEvaluator.create(eval_config)
    
    # 2. Sanity Check: Ensure sign flipping is working
    # FIX 2: Instantiate RealGenome directly, not via config
    test_genome = RealGenome(values=jnp.ones(DIMENSIONS))
    test_fit = evaluator.evaluate(test_genome)
    
    # Sphere(1,1...) > 0. Since maximize=True, output should be NEGATIVE.
    if test_fit > 0:
        print(f"⚠️ WARNING: Evaluator check failed! Fitness is positive ({test_fit}) but maximize=True.")
    
    selection = ElitePoolSelection(num_selections=POP_SIZE, elite_k=ELITE_POOL_SIZE, input_length=POP_SIZE)
    crossover = UniformCrossover(num_offspring=2, crossover_rate =  CROSSOVER_RATE)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH, clip = False)
    
    engine_params = GeneticEngineParams(pop_size=POP_SIZE, num_generations=NUM_GENS_FULL)
    
    engine = GeneticEngine(
        evaluator=evaluator, selection=selection, crossover=crossover, mutation=mutation,
        genome_config=genome_config, engine_params=engine_params
    )
    
    key = jar.PRNGKey(seed)
    state = engine.init_state(key)
    return engine, state

def setup_evosax(seed):
    """Creates Evosax components."""
    rng = jar.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    
    # FIX: Use FIXED_PROBLEM_SEED for Problem Init
    problem = BBOBProblem("sphere", num_dims=DIMENSIONS, seed=FIXED_PROBLEM_SEED)
    # The 'rng' passed to problem.init determines the shift/rotation. 
    # We MUST use a fixed key here to match MalthusJAX's fixed environment.
    param_state = problem.init(jar.PRNGKey(FIXED_PROBLEM_SEED))
    
    strategy = SimpleGA(population_size=POP_SIZE, solution=problem.sample(jar.PRNGKey(0)))
    es_params = strategy.default_params.replace(crossover_rate = CROSSOVER_RATE)
    
    # Init Strategy State
    initial_pop = jax.random.uniform(rng_init, (POP_SIZE, DIMENSIONS), minval=-5.0, maxval=5.0)
    initial_fitness = jnp.full((POP_SIZE,), jnp.inf)
    state = strategy.init(rng, initial_pop, initial_fitness, es_params)
    
    return strategy, problem, es_params, (state, param_state, rng)

# ==========================================
# EXECUTION LOOPS
# ==========================================
def run_evosax_uncompiled(strategy, problem, es_params, carry):
    # Pure Python loop driving JAX ops
    state, p_state, rng = carry
    for _ in range(NUM_GENS_FAST):
        rng, r_a, r_e, r_t = jax.random.split(rng, 4)
        x, state = strategy.ask(r_a, state, es_params)
        fit, p_state, _ = problem.eval(r_e, x, p_state)
        state, _ = strategy.tell(r_t, x, fit, state, es_params)
        
    return state.best_fitness

def run_evosax_scan(strategy, problem, es_params, carry, length):
    # The scan body to be JIT-compiled
    def step(c, _):
        s, p, r = c
        r, r_a, r_e, r_t = jax.random.split(r, 4)
        x, s = strategy.ask(r_a, s, es_params)
        fit, p, _ = problem.eval(r_e, x, p)
        s, _ = strategy.tell(r_t, x, fit, s, es_params)
        return (s, p, r), None

    final_carry, _ = jax.lax.scan(step, carry, None, length=length)
    return final_carry[0].best_fitness

# ==========================================
# MAIN BENCHMARK
# ==========================================
def main():
    print(f"🚀 Benchmarking Comparison (Pop: {POP_SIZE}, Seeds: {NUM_SEEDS})")
    print(f"   Device: {jax.devices()[0].device_kind}")
    print("-" * 60)

    results = []

    for seed in range(NUM_SEEDS):
        print(f"\r  Running Seed {seed+1}/{NUM_SEEDS}...", end="", flush=True)
        
        # -------------------------------------------------
        # 1. MALTHUSJAX
        # -------------------------------------------------
        engine, state = setup_malthus(seed)
        
        # A. Uncompiled (Speed Test Only)
        engine_py = engine.replace(engine_params=engine.engine_params.replace(num_generations=NUM_GENS_FAST))
        t0 = time.time()
        _ = engine_py.run(state, compile=False)
        dt = time.time() - t0
        results.append({
            "Framework": "MalthusJAX", "Mode": "Uncompiled", 
            "Speed": NUM_GENS_FAST/dt, "Final_Cost": None 
        })
        
        # B. Cold Compiled
        engine, state = setup_malthus(seed) # Refresh
        t0 = time.time()
        final, _, _ = engine.run(state, compile=True)
        _ = final.best_fitness.block_until_ready()
        dt = time.time() - t0
        # Metric: Flip sign back to positive cost
        cost = -float(final.best_fitness) 
        results.append({
            "Framework": "MalthusJAX", "Mode": "Cold", 
            "Speed": NUM_GENS_FULL/dt, "Final_Cost": cost
        })
        
        # C. Warm Compiled
        state = engine.init_state(jar.PRNGKey(seed))
        t0 = time.time()
        final, _, _ = engine.run(state, compile=True)
        _ = final.best_fitness.block_until_ready()
        dt = time.time() - t0
        cost = -float(final.best_fitness)
        results.append({
            "Framework": "MalthusJAX", "Mode": "Warm", 
            "Speed": NUM_GENS_FULL/dt, "Final_Cost": cost
        })

        # -------------------------------------------------
        # 2. EVOSAX
        # -------------------------------------------------
        strat, prob, params, carry = setup_evosax(seed)
        
        # A. Uncompiled
        _, _, _, carry_py = setup_evosax(seed)
        t0 = time.time()
        fit = run_evosax_uncompiled(strat, prob, params, carry_py)
        fit.block_until_ready()
        dt = time.time() - t0
        results.append({
            "Framework": "Evosax", "Mode": "Uncompiled", 
            "Speed": NUM_GENS_FAST/dt, "Final_Cost": None
        })

        # Setup JIT Scan
        scan_jit = jax.jit(lambda c: run_evosax_scan(strat, prob, params, c, NUM_GENS_FULL))
        
        # B. Cold Compiled
        _, _, _, carry_cold = setup_evosax(seed)
        t0 = time.time()
        fit = scan_jit(carry_cold)
        fit.block_until_ready()
        dt = time.time() - t0
        results.append({
            "Framework": "Evosax", "Mode": "Cold", 
            "Speed": NUM_GENS_FULL/dt, "Final_Cost": float(fit)
        })
        
        # C. Warm Compiled
        _, _, _, carry_warm = setup_evosax(seed)
        t0 = time.time()
        fit = scan_jit(carry_warm)
        fit.block_until_ready()
        dt = time.time() - t0
        results.append({
            "Framework": "Evosax", "Mode": "Warm", 
            "Speed": NUM_GENS_FULL/dt, "Final_Cost": float(fit)
        })

    print("\n\n✅ Benchmark Complete.")
    df = pd.DataFrame(results)
    
    # ==========================================
    # PLOT 1: SPEED BOXPLOT
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.boxplot(x="Mode", y="Speed", hue="Framework", data=df, showfliers=False, palette="viridis")
    sns.stripplot(x="Mode", y="Speed", hue="Framework", data=df, dodge=True, color="black", alpha=0.5, jitter=True, legend=False)
    
    plt.yscale("log")
    plt.title(f"Speed Comparison: MalthusJAX vs Evosax\n(Pop Size: {POP_SIZE}, 30 Seeds)", fontsize=14, fontweight='bold')
    plt.ylabel("Generations Per Second (Log Scale)", fontsize=12)
    plt.legend(title="Framework")
    
    filename_speed = "benchmark_speed_boxplot.png"
    plt.savefig(filename_speed, dpi=300, bbox_inches='tight')
    print(f"📊 Speed Plot saved to {filename_speed}")
    
    # ==========================================
    # PLOT 2: ACCURACY VIOLIN PLOT
    # ==========================================
    plt.figure(figsize=(8, 6))
    df_warm = df[df["Mode"] == "Warm"]
    
    # Explicit hue assignment to match x
    sns.violinplot(x="Framework", y="Final_Cost", hue="Framework", data=df_warm, palette="muted", inner="quartile", legend=False)
    sns.stripplot(x="Framework", y="Final_Cost", hue="Framework", data=df_warm, color="black", alpha=0.3, jitter=True, legend=False)
    
    plt.title(f"Algorithmic Accuracy Distribution\n(Sphere Problem, 1000 Gens)", fontsize=14, fontweight='bold')
    plt.ylabel("Final Cost (Lower is Better)", fontsize=12)
    plt.xlabel("")
    plt.yscale("log") 
    
    filename_fitness = "benchmark_accuracy_violin.png"
    plt.savefig(filename_fitness, dpi=300, bbox_inches='tight')
    print(f"📊 Accuracy Plot saved to {filename_fitness}")

    # Auto-download for Colab
    try:
        from google.colab import files # type: ignore
        files.download(filename_speed)
        files.download(filename_fitness)
    except ImportError:
        pass

    # Print Summary Table
    print("\nSummary Statistics (Warm Mode):")
    summary = df_warm.groupby("Framework")[["Speed", "Final_Cost"]].describe()
    print(summary)

if __name__ == "__main__":
    main()