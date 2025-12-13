import time
import sys
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jar
import matplotlib.pyplot as plt
import seaborn as sns

# --- Imports ---
from evosax.algorithms.population_based.simple_ga import SimpleGA
from evosax.problems import BBOBProblem
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation

# ==========================================
# DEMO CONFIGURATION
# ==========================================
POPULATION_SIZES = [128, 512, 2048, 8192, 16384]
NUM_SEEDS = 3        
NUM_GENS = 500       
DIMENSIONS = 20

# ==========================================
# SYSTEM UTILS
# ==========================================
def get_environment_info():
    """Detects Colab vs Local and Device Name."""
    # 1. Detect Environment
    if 'google.colab' in sys.modules:
        env_name = "Google Colab"
    else:
        env_name = "Local/Server"
        
    # 2. Detect Hardware
    try:
        device = jax.devices()[0]
        hw_name = f"{device.platform.upper()} ({device.device_kind})"
    except:
        hw_name = "UNKNOWN DEVICE"
        
    return env_name, hw_name

def run_comparison(pop_size, seed):
    # ------------------------------------------
    # 1. MalthusJAX (The Challenger)
    # ------------------------------------------
    # Setup
    genome_config = RealGenomeConfig(length=DIMENSIONS, bounds=(-5.0, 5.0))
    # Note: maximize=True because GeneticEngine usually assumes higher is better
    eval_config = BBOBConfig(fn_name="sphere", num_dims=DIMENSIONS, maximize=True)
    evaluator = BBOBEvaluator.create(eval_config)
    
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=int(pop_size*0.2), input_length=pop_size)
    crossover = UniformCrossover(num_offspring=2, crossover_rate=0.5)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.3, mutation_strength=0.1)
    
    engine = GeneticEngine(
        evaluator=evaluator, selection=selection, crossover=crossover, mutation=mutation,
        genome_config=genome_config,
        engine_params=GeneticEngineParams(pop_size=pop_size, num_generations=NUM_GENS)
    )
    
    # Warmup & Run
    key = jar.PRNGKey(seed)
    state = engine.init_state(key)
    # Warmup
    _ = engine.run(state, compile=True)
    
    # Timed Run
    state = engine.init_state(key) # Reset
    t0 = time.time()
    _, _, _ = engine.run(state, compile=True)
    mjx_time = time.time() - t0
    
    # ------------------------------------------
    # 2. Evosax (The Baseline)
    # ------------------------------------------
    problem = BBOBProblem("sphere", num_dims=DIMENSIONS, seed=0)
    strategy = SimpleGA(population_size=pop_size, solution=problem.sample(jar.PRNGKey(0)))
    es_params = strategy.default_params.replace(crossover_rate=0.5)
    
    # Define Scan Loop
    def step_impl(carry, _):
        state, p_state, rng = carry
        r_a, r_e, r_t, r_next = jax.random.split(rng, 4)
        x, state = strategy.ask(r_a, state, es_params)
        fit, p_state, _ = problem.eval(r_e, x, p_state)
        state, _ = strategy.tell(r_t, x, fit, state, es_params)
        return (state, p_state, r_next), None

    # Init
    rng = jar.PRNGKey(seed)
    carry_init = (
        strategy.init(rng, jnp.zeros((pop_size, DIMENSIONS)), jnp.zeros(pop_size), es_params),
        problem.init(rng),
        rng
    )
    
    # Warmup & Run
    scan_fn = jax.jit(lambda c: jax.lax.scan(step_impl, c, None, length=NUM_GENS))
    _ = scan_fn(carry_init) # Warmup
    
    t0 = time.time()
    _ = scan_fn(carry_init)
    es_time = time.time() - t0
    
    return {
        "Pop_Size": pop_size,
        "MalthusJAX_GensPerSec": NUM_GENS / mjx_time,
        "Evosax_GensPerSec": NUM_GENS / es_time,
        "Speedup": (NUM_GENS / mjx_time) / (NUM_GENS / es_time)
    }

def main():
    env_name, hw_name = get_environment_info()
    
    print(f"🚀 Running Demo Benchmark...")
    print(f"Environment: {env_name}")
    print(f"Hardware:    {hw_name}")
    print(f"Goal:        Prove scaling advantage on this hardware.")
    print("-" * 50)
    
    results = []
    
    for pop in POPULATION_SIZES:
        print(f"  Testing Population: {pop}...", end="", flush=True)
        # Average over seeds
        seed_results = [run_comparison(pop, s) for s in range(NUM_SEEDS)]
        
        avg_speedup = sum(r["Speedup"] for r in seed_results) / NUM_SEEDS
        avg_mjx = sum(r["MalthusJAX_GensPerSec"] for r in seed_results) / NUM_SEEDS
        
        print(f" Done. Speedup: {avg_speedup:.2f}x ({avg_mjx:,.0f} gen/s)")
        
        results.extend(seed_results)

    df = pd.DataFrame(results)
    
    # ==========================================
    # GENERATE SALES PITCH PLOT
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot Throughput
    df_melt = df.melt(id_vars=["Pop_Size"], value_vars=["MalthusJAX_GensPerSec", "Evosax_GensPerSec"], 
                      var_name="Framework", value_name="Throughput")
    
    sns.lineplot(data=df_melt, x="Pop_Size", y="Throughput", hue="Framework", marker="o", linewidth=2.5)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"MalthusJAX Scaling: {env_name}\n({hw_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Population Size (Log Scale)", fontsize=12)
    plt.ylabel("Generations Per Second (Log Scale)", fontsize=12)
    plt.grid(True, which="minor", ls="--", alpha=0.3)
    
    filename = "benchmark_demo_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"\n📊 Plot saved to {filename}")
    
    # ==========================================
    # PRINT THE "ASK"
    # ==========================================
    # Extrapolate for a large run (e.g. 1M generations)
    large_pop_perf = df[df["Pop_Size"] == POPULATION_SIZES[-1]]["MalthusJAX_GensPerSec"].mean()
    baseline_perf = df[df["Pop_Size"] == POPULATION_SIZES[-1]]["Evosax_GensPerSec"].mean()
    
    workload_gens = 1_000_000
    time_mjx = workload_gens / large_pop_perf / 60 # minutes
    time_base = workload_gens / baseline_perf / 60 # minutes

    print("="*50)
    print(f"Context: {env_name} on {hw_name}")
    print(f"For a standard research run ({workload_gens/1e6:.0f}M generations, {POPULATION_SIZES[-1]} pop):")
    print(f"  - Current Method:  {time_base:.1f} minutes")
    print(f"  - MalthusJAX:      {time_mjx:.1f} minutes")
    
    speedup = large_pop_perf / baseline_perf
    print(f"  - Performance:     {speedup:.1f}x Faster")
    print("="*50)

if __name__ == "__main__":
    main()