import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy import stats

# --- Framework Imports (Assumes these are installed) ---
try:
    import malthusjax as mjx
    from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
    from evosax.problems import BBOBProblem
    from evosax.algorithms import SimpleGA
except ImportError as e:
    print("❌ Critical Dependency Missing. Please install malthusjax and evosax.")
    raise e

# Suppress JAX/TensorFlow warnings for clean output
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ENGINE BUILDERS (Setup Logic)
# ==============================================================================

def setup_bbob_problem(problem_name, dim, seed):
    """Creates the objective function for both frameworks."""
    # MalthusJAX
    bbob_config = BBOBConfig(fn_name=problem_name, num_dims=dim, seed=seed, maximize=True)
    mjx_evaluator = BBOBEvaluator.create(bbob_config)
    
    # EvoSax
    es_problem = BBOBProblem(problem_name, num_dims=dim, seed=seed)
    return mjx_evaluator, es_problem

def build_malthusjax(evaluator, pop_size, num_gen, crossover_rate=0.5):
    """Builds the MalthusJAX Genetic Engine."""
    genome_config = mjx.RealGenomeConfig(length=evaluator.config.num_dims, bounds=(-5.0, 5.0))
    params = mjx.AbstractEngineParams(pop_size=pop_size, num_generations=num_gen, elitism=1) # Enable Elitism!

    # Operators
    selection = mjx.selection.ElitePool(num_selections=pop_size, elite_k=int(pop_size * 0.5))
    crossover = mjx.crossover.realUniform(num_offspring=2, crossover_rate=crossover_rate) # Configurable
    mutation = mjx.mutation.Gaussian(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1) # 2 keys fixed

    return mjx.GeneticEngine(
        genome_config=genome_config, evaluator=evaluator, selection=selection,
        crossover=crossover, mutation=mutation, engine_params=params
    )

def build_evosax(problem, pop_size, crossover_rate=0.0):
    """Builds the EvoSax Strategy."""
    rng = jax.random.PRNGKey(0)
    init_sol = problem.sample(rng)
    strategy = SimpleGA(population_size=pop_size, 
                        solution=init_sol)
    strategy.elite_ratio = 0.5
    # Inject params
    es_params = strategy.default_params.replace(crossover_rate=crossover_rate)
    return strategy, es_params

# ==============================================================================
# 2. RUNNERS (Execution & Timing)
# ==============================================================================

def run_single_experiment(framework, problem_name, pop_size, dim=20, gens=50, seed=42):
    """
    Executes one standardized run with correct JAX blocking and timing.
    """
    mjx_eval, es_prob = setup_bbob_problem(problem_name, dim, seed)
    key = jax.random.PRNGKey(seed)
    
    if framework == "MalthusJAX":
        # Configure Engine (Enable Crossover=0.5 for GA logic)
        engine = build_malthusjax(mjx_eval, pop_size, gens, crossover_rate=0.5)
        
        # JIT Compiled Step Function
        @jax.jit
        def run_loop(state):
            return engine.run(state)
        
        # Initialization
        state = engine.init_state(key)
        
        # Warmup (Compile)
        _ = run_loop(state) 
        
        # Measurement
        state = engine.init_state(key) # Reset
        start = time.time()
        final_state, _, _ = run_loop(state)
        jax.block_until_ready(final_state.best_fitness) # CRITICAL: Block Async
        runtime = time.time() - start
        
        return -float(final_state.best_fitness), runtime # Flip sign (Minimization)

    elif framework == "EvoSax":
        # Configure Strategy (Default Crossover=0.0 for ES logic)
        strategy, params = build_evosax(es_prob, pop_size, crossover_rate=0.0)
        
        @jax.jit
        def run_loop(rng, state, p_state):
            # EvoSax Scan Loop
            def step(carry, _):
                s, ps, r = carry
                r, r_step = jax.random.split(r)
                x, s = strategy.ask(r, s, params)
                fit, ps, _ = es_prob.eval(r, x, ps)
                s, _ = strategy.tell(r, x, fit, s, params)
                return (s, ps, r), None
            
            final, _ = jax.lax.scan(step, (state, p_state, rng), None, length=gens)
            return final[0]

        # Initialization
        r_init, r_run = jax.random.split(key)
        r_init, r_ask = jax.random.split(r_init)
        r_init, r_prob = jax.random.split(r_init)
        p_state = es_prob.init(r_prob)
        
        # Initialize population and get fitness
        init_pop = jax.random.uniform(r_ask, (pop_size, dim), minval=-5.0, maxval=5.0)
        init_fit, p_state, _ = es_prob.eval(r_init, init_pop, p_state)
        state = strategy.init(r_init, init_pop, init_fit, params)
        
        # Warmup
        _ = run_loop(r_run, state, p_state)
        
        # Measurement
        start = time.time()
        final_state = run_loop(r_run, state, p_state)
        jax.block_until_ready(final_state.best_fitness) # CRITICAL: Block Async
        runtime = time.time() - start
        
        return float(final_state.best_fitness), runtime

# ==============================================================================
# 3. SCORECARD ENGINE
# ==============================================================================

class BenchmarkScorecard:
    def __init__(self, show_plots=False):
        self.show_plots = show_plots
        self.scaling_pops = [100, 1000, 10000] # Log Scale Populations
        self.test_funcs = ["sphere", "rastrigin"] # Representative Subset
        
    def measure_speed(self):
        print("\n🏎️  MEASURING SPEED & SCALING (Objective 1)")
        print("-" * 60)
        results = []
        
        for pop in self.scaling_pops:
            print(f"   Testing Population N={pop}...", end="", flush=True)
            
            # Run Sphere (Cheap eval to isolate engine overhead)
            _, t_mjx = run_single_experiment("MalthusJAX", "sphere", pop, gens=100)
            _, t_es = run_single_experiment("EvoSax", "sphere", pop, gens=100)
            
            # Calculate Throughput (Evals / Sec)
            # Evals = Pop * Gens
            tp_mjx = (pop * 100) / t_mjx
            tp_es = (pop * 100) / t_es
            
            print(f" Done. (MJX: {tp_mjx/1e6:.2f}M/s | ES: {tp_es/1e6:.2f}M/s)")
            
            results.append({
                "pop": pop,
                "mjx_throughput": tp_mjx,
                "es_throughput": tp_es,
                "mjx_time": t_mjx,
                "es_time": t_es
            })
            
        return pd.DataFrame(results)

    def measure_accuracy(self):
        print("\n🎯 MEASURING ACCURACY & ROBUSTNESS (Objective 2)")
        print("-" * 60)
        results = []
        
        # Pop=256 is standard for accuracy benchmarks
        for func in self.test_funcs:
            print(f"   Testing Function '{func}'...", end="", flush=True)
            
            # Use fixed seeds for reproducibility
            c_mjx, _ = run_single_experiment("MalthusJAX", func, pop_size=256, gens=100)
            c_es, _ = run_single_experiment("EvoSax", func, pop_size=256, gens=100)
            
            print(f" Cost: MJX={c_mjx:.2f} | ES={c_es:.2f}")
            results.append({"problem": func, "mjx_cost": c_mjx, "es_cost": c_es})
            
        return pd.DataFrame(results)

    def capture_trace(self):
        print("\n📸 CAPTURING JAX TRACE (Objective 3)")
        print("-" * 60)
        trace_path = "/tmp/jax_trace"
        
        # Setup trace run
        print(f"   Running heavy workload (N=10k) for tracing...")
        jax.profiler.start_trace(trace_path)
        
        # Run one iteration of large pop
        run_single_experiment("MalthusJAX", "sphere", pop_size=10000, gens=10)
        
        jax.profiler.stop_trace()
        print(f"   ✓ Trace saved to {trace_path}")
        print("   (Upload this folder to ui.perfetto.dev to calculate Fusion Index)")

    def report(self, speed_df, acc_df):
        print("\n" + "="*60)
        print("🏆 FINAL BENCHMARK SCORECARD")
        print("="*60)
        
        # 1. Speed Score (Peak Throughput Ratio)
        peak_mjx = speed_df['mjx_throughput'].max()
        peak_es = speed_df['es_throughput'].max()
        speed_score = peak_mjx / peak_es
        
        # 2. Latency Penalty (Low Pop)
        lat_mjx = speed_df.iloc[0]['mjx_time']
        lat_es = speed_df.iloc[0]['es_time']
        latency_score = lat_es / lat_mjx # Higher is better (less penalty)
        
        # 3. Accuracy Gap (Geo Mean)
        # Simple ratio for now
        acc_score = acc_df['es_cost'].mean() / acc_df['mjx_cost'].mean()
        
        print(f"\n1. PEAK THROUGHPUT SCORE:  {speed_score:.2f}x (vs Reference)")
        print(f"   (Higher > 1.0 means superior vectorization scaling)")
        
        print(f"\n2. LATENCY EFFICIENCY:     {latency_score:.2f}x")
        print(f"   (Lower < 1.0 means higher CPU overhead/setup cost)")
        
        print(f"\n3. ACCURACY PARITY:        {acc_score:.2f}")
        print(f"   (Near 1.0 means algorithmic equivalence)")

        if self.show_plots:
            self.plot(speed_df, acc_df)

    def plot(self, speed_df, acc_df):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speed Scaling Plot
        ax1.plot(speed_df['pop'], speed_df['mjx_throughput'], 'b-o', label='MalthusJAX')
        ax1.plot(speed_df['pop'], speed_df['es_throughput'], 'r--s', label='EvoSax')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title("Throughput Scaling (Evals/Sec)")
        ax1.set_xlabel("Population Size")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy Bar Chart
        x = np.arange(len(acc_df))
        width = 0.35
        ax2.bar(x - width/2, acc_df['mjx_cost'], width, label='MalthusJAX')
        ax2.bar(x + width/2, acc_df['es_cost'], width, label='EvoSax')
        ax2.set_xticks(x)
        ax2.set_xticklabels(acc_df['problem'])
        ax2.set_title("Solution Quality (Lower is Better)")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalthusJAX vs EvoSax Benchmark")
    parser.add_argument("--plot", action="store_true", help="Show performance graphs")
    args = parser.parse_args()
    
    # Check Hardware
    print(f"HARDWARE DETECTED: {jax.devices()}")
    
    # Run Suite
    scorecard = BenchmarkScorecard(show_plots=args.plot)
    
    speed_data = scorecard.measure_speed()
    acc_data = scorecard.measure_accuracy()
    scorecard.capture_trace()
    
    scorecard.report(speed_data, acc_data)