import time
import jax
from dataclasses import dataclass
from .adapters import AbstractBenchmarkAdapter

@dataclass
class BenchmarkResult:
    framework: str        # "MalthusJAX" or "Evosax"
    device: str           # "NVIDIA A100"
    compile_time: float   # Seconds
    execution_time: float # Seconds
    generations_per_sec: float
    best_fitness: float
    fitness_std: float = 0.0  # Standard deviation across repeats

def run_adapter_benchmark(
    adapter: AbstractBenchmarkAdapter, 
    num_gens: int, 
    seed: int,
    framework_name: str,
    unroll: int = 1,
    repeats: int = 1,
) -> BenchmarkResult:
    """
    Standardized JIT + Run + Time routine.
    """
    rng = jax.random.PRNGKey(seed)
    
    # 1. Initialization
    init_carry = adapter.init(rng)
    
    # 2. Get the Step Function
    step_fn = adapter.make_step_fn()
    
    # 3. Define the Scan Loop (honor the unroll factor)
    def scan_loop(carry):
        # pass `unroll` into lax.scan to allow loop unrolling where supported
        return jax.lax.scan(step_fn, carry, None, length=num_gens, unroll=unroll)
    
    # 4. Compilation (Warmup)
    print(f"[{framework_name}] Compiling...", end="", flush=True)
    jit_scan = jax.jit(scan_loop)
    
    t0 = time.perf_counter()
    # Force compilation by running it and blocking
    compiled_scan = jit_scan.lower(init_carry).compile()
    t_compile = time.perf_counter() - t0
    print(f" Done ({t_compile:.4f}s)")
    
    # 5. Execution (Hot Run)
    # Note: We run the compiled executable directly. Execute `repeats` times
    # and average the execution time for more stable measurements.
    total_exec_time = 0.0
    final_carry = None
    fitness_values = []
    
    for i in range(max(1, repeats)):
        t0 = time.perf_counter()
        final_carry, _ = compiled_scan(init_carry)
        # Block on the result to ensure device sync
        fitness_scalar = adapter.get_best_fitness(final_carry)
        _ = jax.block_until_ready(jax.numpy.array(fitness_scalar))
        t_run = time.perf_counter() - t0
        total_exec_time += t_run
        fitness_values.append(fitness_scalar)

    avg_exec = total_exec_time / max(1, repeats)
    
    # Calculate fitness statistics
    import numpy as np
    fitness_mean = float(np.mean(fitness_values))
    fitness_std = float(np.std(fitness_values)) if len(fitness_values) > 1 else 0.0
    
    return BenchmarkResult(
        framework=framework_name,
        device=adapter.get_device_info(),
        compile_time=t_compile,
        execution_time=avg_exec,
        generations_per_sec=num_gens / avg_exec,
        best_fitness=fitness_mean,
        fitness_std=fitness_std,
    )