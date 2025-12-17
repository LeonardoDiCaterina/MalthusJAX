import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy import stats

# --- Framework Imports ---
try:
    import malthusjax as mjx
    from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
    from evosax.problems import BBOBProblem
    from evosax.algorithms import SimpleGA
except ImportError as e:
    print("Critical Dependency Missing. Please install malthusjax and evosax.")
    raise e

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SETUP & BUILDERS
# ==============================================================================

def setup_bbob_problem(problem_name, dim, seed):
    """Creates the objective function for both frameworks."""
    # Malthus uses Maximization by default, so we flip BBOB (min) to max
    bbob_config = BBOBConfig(fn_name=problem_name, num_dims=dim, seed=seed, maximize=True)
    mjx_evaluator = BBOBEvaluator.create(bbob_config)
    
    # EvoSax - minimization by default
    es_problem = BBOBProblem(problem_name, num_dims=dim, seed=seed)
    return mjx_evaluator, es_problem

def build_malthusjax(evaluator, pop_size, num_gen):
    """Builds the MalthusJAX Genetic Engine."""
    genome_config = mjx.RealGenomeConfig(length=evaluator.config.num_dims, bounds=(-5.0, 5.0))
    # We set num_generations=1 because we will control the loop manually in the runner
    params = mjx.AbstractEngineParams(pop_size=pop_size, num_generations=1, elitism=0)

    selection = mjx.selection.ElitePool(num_selections=pop_size, elite_k=int(pop_size * 0.5))
    crossover = mjx.crossover.realUniform(num_offspring=2, crossover_rate=0.5)
    mutation = mjx.mutation.Gaussian(num_offspring=1, mutation_rate=1.0, mutation_strength=1.0, clip=False)

    return mjx.GeneticEngine(
        genome_config=genome_config, evaluator=evaluator, selection=selection,
        crossover=crossover, mutation=mutation, engine_params=params
    )

def build_evosax(problem, pop_size):
    """Builds the EvoSax Strategy."""
    rng = jax.random.PRNGKey(0)
    init_sol = problem.sample(rng)
    strategy = SimpleGA(population_size=pop_size, solution=init_sol)
    strategy.elite_ratio = 0.5
    es_params = strategy.default_params.replace(crossover_rate=0.5)
    return strategy, es_params

# ==============================================================================
# 2. STATISTICAL RUNNER (The Core Upgrade)
# ==============================================================================

def run_batch_experiment(framework, problem_name, pop_size, unroll_factor, dim=20, gens=1000, repeats=30, seed=42):
    """
    Runs the experiment N times and returns (mean_time, std_time, throughput).
    Manually controls 'scan' to enforce 'unroll_factor'.
    """
    mjx_eval, es_prob = setup_bbob_problem(problem_name, dim, seed)
    key = jax.random.PRNGKey(seed)
    
    # --- MALTHUS JAX RUNNER ---
    if framework == "MalthusJAX":
        engine = build_malthusjax(mjx_eval, pop_size, gens)
        
        # We manually expose the step function to control unrolling
        def mjx_scan_body(carry, _):
            state = carry
            # Engine.step returns (new_state, metrics)
            new_state, metrics = engine.step(state)
            return new_state, metrics

        @jax.jit
        def run_loop(state):
            # THE UNROLL TICK:
            final_state, _ = jax.lax.scan(mjx_scan_body, state, None, length=gens, unroll=unroll_factor)
            return final_state

        # Init & Warmup
        state = engine.init_state(key)
        _ = run_loop(state).best_fitness.block_until_ready()
        
        times = []
        for i in range(repeats):
            # Reset state for each run with a new key variant
            iter_key = jax.random.fold_in(key, i)
            state = engine.init_state(iter_key)
            
            # Hot Loop
            start = time.perf_counter()
            final_state = run_loop(state)
            _ = final_state.best_fitness.block_until_ready() # Block
            end = time.perf_counter()
            
            times.append(end - start)

    # --- EVOSAX RUNNER ---
    elif framework == "EvoSax":
        strategy, params = build_evosax(es_prob, pop_size)
        
        def es_scan_body(carry, _):
            s, ps, r = carry
            r, r_step = jax.random.split(r)
            x, s = strategy.ask(r, s, params)
            fit, ps, _ = es_prob.eval(r, x, ps)
            s, _ = strategy.tell(r, x, fit, s, params)
            return (s, ps, r), None

        @jax.jit
        def run_loop(rng, state, p_state):
            # THE UNROLL TICK:
            final, _ = jax.lax.scan(es_scan_body, (state, p_state, rng), None, length=gens, unroll=unroll_factor)
            return final[0] # Return state

        # Init & Warmup
        r_init, r_run = jax.random.split(key)
        init_pop = jax.random.uniform(r_init, (pop_size, dim), minval=-5.0, maxval=5.0)
        # Hacky init logic for Evosax
        p_state = es_prob.init(jax.random.PRNGKey(0))
        init_fit, p_state, _ = es_prob.eval(r_init, init_pop, p_state)
        state = strategy.init(r_init, init_pop, init_fit, params)
        
        _ = run_loop(r_run, state, p_state).best_fitness.block_until_ready()

        times = []
        for i in range(repeats):
            iter_key = jax.random.fold_in(key, i)
            r_run = jax.random.split(iter_key)[0]
            
            # Reset
            # Note: We reuse the initial state structure for speed, just changing RNG
            # Ideally we'd re-init completely, but this measures throughput accurately
            
            start = time.perf_counter()
            final_state = run_loop(r_run, state, p_state)
            _ = final_state.best_fitness.block_until_ready()
            end = time.perf_counter()
            
            times.append(end - start)

    # Statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Evals per second
    total_evals = pop_size * gens
    gps = gens / mean_time
    throughput = total_evals / mean_time
    
    return mean_time, std_time, throughput, gps

# ==============================================================================
# 3. THE "SWEET SPOT" SEARCH
# ==============================================================================

def grid_search_unroll():
    # TEST CONFIGURATION
    POP_SIZE = 10000        # The scale where you saw parity/loss on H100
    GENS = 2000             # Enough gens to allow unrolling to work
    UNROLL_LEVELS = [1, 10, 50, 100]
    REPEATS = 30
    
    print(f"\n🔬 STARTING DEEP PROBE (N={POP_SIZE}, Gens={GENS}, Repeats={REPEATS})")
    print(f"HARDWARE: {jax.devices()[0].device_kind}")
    print("=" * 80)
    print(f"{'Unroll':<8} | {'Framework':<12} | {'Time (s)':<12} | {'GPS':<10} | {'Speedup'}")
    print("-" * 80)

    results = []

    for u in UNROLL_LEVELS:
        # Run MalthusJAX
        m_time, m_std, m_tput, m_gps = run_batch_experiment(
            "MalthusJAX", "sphere", POP_SIZE, unroll_factor=u, gens=GENS, repeats=REPEATS
        )
        
        # Run EvoSax
        e_time, e_std, e_tput, e_gps = run_batch_experiment(
            "EvoSax", "sphere", POP_SIZE, unroll_factor=u, gens=GENS, repeats=REPEATS
        )
        
        speedup = m_gps / e_gps
        
        print(f"{u:<8} | {'MalthusJAX':<12} | {m_time:.4f} ±{m_std:.3f} | {m_gps:.0f}       | {speedup:.2f}x")
        print(f"{'':<8} | {'EvoSax':<12}     | {e_time:.4f} ±{e_std:.3f} | {e_gps:.0f}       |")
        print("-" * 80)
        
        results.append({
            "unroll": u,
            "mjx_gps": m_gps,
            "es_gps": e_gps,
            "speedup": speedup
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = grid_search_unroll()
    
    print("\nFINAL SUMMARY")
    print(df)
    
    best_unroll = df.loc[df['speedup'].idxmax()]
    print(f"\nBest Configuration: Unroll={best_unroll['unroll']} (Speedup: {best_unroll['speedup']:.2f}x)")