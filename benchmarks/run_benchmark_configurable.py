import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy import stats
import json
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns 
import gc # Added for memory cleanup
from functools import partial

# --- FRAMEWORK IMPORTS ---
try:
    import malthusjax as mjx
    from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
    from evosax.problems import BBOBProblem
    from evosax.algorithms import SimpleGA
except ImportError as e:
    print(f"❌ Critical Dependency Missing: {e}")
    raise e

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def setup_bbob_problem(problem_name, dim, seed):
    """Creates the objective function for both frameworks."""
    bbob_config = BBOBConfig(fn_name=problem_name, num_dims=dim, seed=seed, maximize=True)
    mjx_evaluator = BBOBEvaluator.create(bbob_config)
    es_problem = BBOBProblem(problem_name, num_dims=dim, seed=seed)
    return mjx_evaluator, es_problem

def build_malthusjax(evaluator, pop_size, num_gen, crossover_rate=0.5):
    """Builds the MalthusJAX Genetic Engine."""
    genome_config = mjx.RealGenomeConfig(length=evaluator.config.num_dims, bounds=(-5.0, 5.0))
    params = mjx.AbstractEngineParams(pop_size=pop_size, num_generations=num_gen, elitism=1) 
    selection = mjx.selection.ElitePool(num_selections=pop_size, elite_k=int(pop_size * 0.5))
    crossover = mjx.crossover.realUniform(num_offspring=2, crossover_rate=crossover_rate)
    mutation = mjx.mutation.Gaussian(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1)
    return mjx.GeneticEngine(
        genome_config=genome_config, evaluator=evaluator, selection=selection,
        crossover=crossover, mutation=mutation, engine_params=params
    )

def build_evosax(problem, pop_size, crossover_rate=0.0):
    """Builds the EvoSax Strategy."""
    rng = jax.random.PRNGKey(0)
    init_sol = problem.sample(rng)
    strategy = SimpleGA(population_size=pop_size, solution=init_sol)
    strategy.elite_ratio = 0.5
    es_params = strategy.default_params.replace(crossover_rate=crossover_rate)
    return strategy, es_params

# ==============================================================================
# 2. RUNNERS (Single Experiment Logic)
# ==============================================================================

def run_single_experiment(framework, problem_name, pop_size, dim, gens, seed, crossover_rate):
    """Executes one standardized run with correct JAX blocking and timing."""
    mjx_eval, es_prob = setup_bbob_problem(problem_name, dim, seed)
    key = jax.random.PRNGKey(seed)
    
    if framework == "MalthusJAX":
        engine = build_malthusjax(mjx_eval, pop_size, gens, crossover_rate=crossover_rate)
        
        @jax.jit
        def run_loop(state): return engine.run(state)
        
        state = engine.init_state(key)
        _ = run_loop(state) # Warmup (Compilation)
        state = engine.init_state(key) # Reset state for timing
        start = time.time()
        final_state, _, _ = run_loop(state)
        jax.block_until_ready(final_state.best_fitness) # Critical Sync
        runtime = time.time() - start
        
        # Cleanup after MalthusJAX run
        del engine, state, final_state
        gc.collect()
        
        return -float(final_state.best_fitness), runtime # Flip sign (Minimization)

    elif framework == "EvoSax":
        strategy, params = build_evosax(es_prob, pop_size, crossover_rate=0.0) 
        
        @jax.jit
        def run_loop(rng, state, p_state):
            def step(carry, _):
                s, ps, r = carry
                r, r_step = jax.random.split(r)
                x, s = strategy.ask(r, s, params)
                fit, ps, _ = es_prob.eval(r, x, ps)
                s, _ = strategy.tell(r, x, fit, s, params)
                return (s, ps, r), None
            
            final, _ = jax.lax.scan(step, (state, p_state, rng), None, length=gens)
            return final[0]

        r_init, r_run = jax.random.split(key)
        r_init, r_ask = jax.random.split(r_init)
        r_init, r_prob = jax.random.split(r_init)
        p_state = es_prob.init(r_prob)
        
        init_pop = jax.random.uniform(r_ask, (pop_size, dim), minval=-5.0, maxval=5.0)
        init_fit, p_state, _ = es_prob.eval(r_init, init_pop, p_state)
        state = strategy.init(r_init, init_pop, init_fit, params)

        _ = run_loop(r_run, state, p_state) # Warmup
        start = time.time()
        final_state = run_loop(r_run, state, p_state)
        jax.block_until_ready(final_state.best_fitness) # Critical Sync
        runtime = time.time() - start
        
        # Cleanup after EvoSax run
        del strategy, state, final_state, p_state
        gc.collect()
        
        return float(final_state.best_fitness), runtime

# ==============================================================================
# 3. SCORECARD ENGINE (JSON, CSV, & Plotting)
# ==============================================================================

class BenchmarkScorecard:
    def __init__(self, config, device_type, show_plots): 
        self.config = config
        self.show_plots = show_plots
        self.all_raw_results = []
        self.benchmark_params = config['benchmark_params']
        self.test_configs = config['test_configurations']
        
        self.device_type = device_type
        self.output_prefix = self.benchmark_params.get('output_prefix', 'bbob_report')
        self.base_filename_prefix = f"{self.output_prefix}_{self.device_type}"
        
        os.makedirs('benchmark_plots', exist_ok=True)
        
    def _cleanup_memory(self):
        """Forces the JAX runtime to clean up device memory."""
        # JAX's internal memory allocator often holds onto memory for future use.
        # This function signals to release it, though JAX doesn't offer a direct "free all" command.
        jax.clear_caches()
        # Explicit garbage collection is the best way to release Python objects referencing device memory
        gc.collect()
        
    def run_suite(self):
        print("\n🚀 STARTING COMPREHENSIVE BBOB BENCHMARK")
        print("=" * 80)
        
        # ... (same logic for checking sphere baseline) ...
        def check_sphere_baseline():
            for config_set in self.test_configs:
                if config_set['problem'] == 'sphere' and 0.0 in config_set['crossover_rates']:
                    return True
            return False
        
        if not check_sphere_baseline():
            print("Warning: Adding 'sphere' baseline test (Cr=0.0) for speed calculation.")
            self.test_configs.append({"problem": "sphere", "dimensions": [20], "crossover_rates": [0.0], "pop_sizes": [100, 1000, 10000]})

        
        for config_set in self.test_configs:
            problem = config_set['problem']
            for dim in config_set['dimensions']:
                for pop in config_set['pop_sizes']:
                    for rate in config_set['crossover_rates']:
                        
                        # Use a fixed run count from benchmark_params
                        for run_id in range(self.benchmark_params['runs_per_combination']):
                            
                            # --- MEMORY CLEANUP BEFORE EACH TRIAL ---
                            self._cleanup_memory()
                            
                            seed = self.benchmark_params['base_seed'] + run_id
                            
                            print(f"   [RUN {run_id+1}/{self.benchmark_params['runs_per_combination']}] {problem} (D={dim}, N={pop}, Cr={rate})", end="", flush=True)

                            # 1. MJX Run (GA Mode)
                            c_mjx, t_mjx = run_single_experiment("MalthusJAX", problem, pop, dim, self.benchmark_params['generations'], seed, crossover_rate=rate)
                            
                            # 2. EvoSax Run (ES Baseline)
                            c_es, t_es = run_single_experiment("EvoSax", problem, pop, dim, self.benchmark_params['generations'], seed, crossover_rate=0.0)
                            
                            tp_mjx = (pop * self.benchmark_params['generations']) / t_mjx
                            tp_es = (pop * self.benchmark_params['generations']) / t_es
                            
                            # Store Raw Result
                            self.all_raw_results.append({
                                "problem": problem, "dim": dim, "pop": pop, "crossover_rate": rate, "seed": seed,
                                "mjx_cost": float(c_mjx), "es_cost": float(c_es),
                                "mjx_runtime": float(t_mjx), "es_runtime": float(t_es),
                                "mjx_throughput": float(tp_mjx), "es_throughput": float(tp_es)
                            })
                            print(" ✅ Done.")
                            
        print("\n✅ ALL EXPERIMENTS COMPLETE.")
        
    def calculate_final_scores(self, df_raw):
        """Calculates aggregated scores for reporting."""
        
        # 1. Throughput & Latency Scores (Fixed problem=sphere, Cr=0.0)
        df_speed = df_raw[(df_raw['problem'] == 'sphere') & (df_raw['crossover_rate'] == 0.0)]
        
        if df_speed.empty:
             return {"error": "Critical: Sphere baseline missing. Check config."}

        # Peak Throughput (Max Pop)
        max_pop = df_speed['pop'].max()
        df_peak = df_speed[df_speed['pop'] == max_pop]
        peak_throughput_ratio = df_peak['mjx_throughput'].mean() / df_peak['es_throughput'].mean()
        
        # Latency Efficiency (Min Pop)
        min_pop = df_speed['pop'].min()
        df_latency = df_speed[df_speed['pop'] == min_pop]
        latency_efficiency = df_latency['es_runtime'].mean() / df_latency['mjx_runtime'].mean()
        
        # 2. Accuracy Scores (Group by problem and MJX Crossover Rate)
        df_acc = df_raw.groupby(['problem', 'crossover_rate', 'dim']).agg(
            mean_mjx_cost=('mjx_cost', 'mean'),
            mean_es_cost=('es_cost', 'mean'),
            std_mjx_cost=('mjx_cost', 'std')
        ).reset_index()
        
        df_acc['accuracy_ratio'] = df_acc['mean_es_cost'] / df_acc['mean_mjx_cost']

        return {
            "summary_metrics": {
                "peak_throughput_ratio": float(peak_throughput_ratio),
                "latency_efficiency_ratio": float(latency_efficiency),
            },
            "accuracy_details": df_acc.to_dict('records')
        }

    def generate_report(self):
        """Prints the final summary and saves raw data to JSON and CSV."""
        if not self.all_raw_results:
            print("No results generated.")
            return

        df_raw = pd.DataFrame(self.all_raw_results)
        final_scores = self.calculate_final_scores(df_raw)
        
        # --- Group Problems for Filename ---
        problem_list = sorted(df_raw['problem'].unique())
        problems_slug = '_'.join(problem_list)
        
        # --- Save Raw Data to Spreadsheet (.csv) ---
        csv_filename = f"{self.base_filename_prefix}_{problems_slug}_raw_data.csv"
        df_raw.to_csv(csv_filename, index=False)
        print(f"\n💾 RAW DATA SAVED TO SPREADSHEET: {csv_filename}")
        
        # --- Save Final Report (JSON) ---
        json_filename = f"{self.base_filename_prefix}_{problems_slug}_report.json"
        output_data = {
            "config": self.config,
            "summary": final_scores['summary_metrics'],
            "accuracy_details": final_scores['accuracy_details'],
        }
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"💾 FINAL REPORT SAVED TO: {json_filename}")

        # --- Print Scorecard ---
        print("\n" + "="*60)
        print("🏆 FINAL BENCHMARK SCORECARD")
        print("="*60)
        
        summary = final_scores['summary_metrics']
        
        print(f"\n1. PEAK THROUGHPUT SCORE:  {summary['peak_throughput_ratio']:.2f}x (vs Reference)")
        print(f"2. LATENCY EFFICIENCY:     {summary['latency_efficiency_ratio']:.2f}x")
        
        print("\n3. ACCURACY SUITE RESULTS (MJX Cost / ES Cost Ratio):")
        
        # Group by Problem and Crossover Rate for printing
        acc_summary = pd.DataFrame(final_scores['accuracy_details'])
        acc_summary['rel_cost_display'] = acc_summary['accuracy_ratio'].apply(lambda x: f"{x:.2f}x")
        
        print(acc_summary[['problem', 'dim', 'crossover_rate', 'accuracy_ratio', 'std_mjx_cost']].to_string(index=False))

    # --- PLOTTING FUNCTIONS ---
    def plot_all(self, df_raw):
        # ... (Plotting imports are at the top)
        
        if not self.show_plots:
             plt.switch_backend('Agg')

        # --- PLOT 1: SCALING ---
        df_speed = df_raw[(df_raw['problem'] == 'sphere') & (df_raw['crossover_rate'] == 0.0)]
        plt.figure(figsize=(8, 6))
        
        # Calculate mean/median for plotting robustness
        df_speed_agg = df_speed.groupby('pop').agg(
            mean_mjx=('mjx_throughput', 'mean'),
            mean_es=('es_throughput', 'mean')
        ).reset_index()

        plt.plot(df_speed_agg['pop'], df_speed_agg['mean_mjx']/1e6, 'b-o', label='MalthusJAX (Mean)')
        plt.plot(df_speed_agg['pop'], df_speed_agg['mean_es']/1e6, 'r--s', label='Evosax (Mean)')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Throughput Scaling: MJX vs ES on Sphere ({self.device_type.upper()})")
        plt.xlabel("Population Size (Log Scale)")
        plt.ylabel("Throughput (Million Evals/Sec)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(f'benchmark_plots/{self.base_filename_prefix}_throughput_scaling.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # --- PLOT 2: ACCURACY BY PROBLEM/RATE ---
        df_plot_acc = df_raw[df_raw['problem'] != 'sphere'] 
        
        if df_plot_acc.empty:
            return

        plt.figure(figsize=(12, 6))
        
        sns.boxplot(data=df_plot_acc, x='problem', y='mjx_cost', hue='crossover_rate', 
                    palette='viridis', dodge=True)
        
        es_baseline_costs = df_plot_acc.groupby('problem')['es_cost'].mean().reset_index()
        
        for index, row in es_baseline_costs.iterrows():
            plt.hlines(row['es_cost'], 
                       xmin=index - 0.45, xmax=index + 0.45, 
                       colors='grey', linestyles='dotted', label='ES Baseline' if index == 0 else "")

        plt.title(f"Solution Quality vs. Crossover Rate ({self.device_type.upper()})")
        plt.ylabel("Mean Best Cost (Lower is Better)")
        plt.xlabel("BBOB Problem")
        plt.legend(title="MJX Cr Rate")
        plt.grid(True, axis='y', alpha=0.5)
        plt.savefig(f'benchmark_plots/{self.base_filename_prefix}_accuracy_by_rate.png', bbox_inches='tight', dpi=300)
        plt.close()


    def run_full_suite(self):
        self.run_suite()
        self.generate_report()
        
        if self.show_plots:
            self.plot_all(pd.DataFrame(self.all_raw_results))


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser(description="MalthusJAX vs Evosax Comprehensive BBOB Benchmark.")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file (e.g., config.json).")
    parser.add_argument("--plot", action="store_true", help="Save all comparison plots to the 'benchmark_plots' directory.")
    
    args = parser.parse_args()

    # --- JAX Device Detection ---
    try:
        device = jax.devices()[0].platform
    except (RuntimeError, IndexError):
        device = 'cpu'
    
    device_slug = device.lower() 

    # --- Load Configuration ---
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        exit(1)

    print(f"HARDWARE DETECTED: {jax.devices()}")

    # --- Execute Suite ---
    scorecard = BenchmarkScorecard(config=config, device_type=device_slug, show_plots=args.plot)
    scorecard.run_full_suite()