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

def run_adapter_benchmark(
    adapter: AbstractBenchmarkAdapter, 
    num_gens: int, 
    seed: int,
    framework_name: str
) -> BenchmarkResult:
    """
    Standardized JIT + Run + Time routine.
    """
    rng = jax.random.PRNGKey(seed)
    
    # 1. Initialization
    init_carry = adapter.init(rng)
    
    # 2. Get the Step Function
    step_fn = adapter.make_step_fn()
    
    # 3. Define the Scan Loop
    def scan_loop(carry):
        return jax.lax.scan(step_fn, carry, None, length=num_gens)
    
    # 4. Compilation (Warmup)
    print(f"[{framework_name}] Compiling...", end="", flush=True)
    jit_scan = jax.jit(scan_loop)
    
    t0 = time.perf_counter()
    # Force compilation by running it and blocking
    compiled_scan = jit_scan.lower(init_carry).compile()
    t_compile = time.perf_counter() - t0
    print(f" Done ({t_compile:.4f}s)")
    
    # 5. Execution (Hot Run)
    # Note: We run the compiled executable directly
    t0 = time.perf_counter()
    final_carry, _ = compiled_scan(init_carry)
    # Block on the result to ensure GPU sync
    fitness_scalar = adapter.get_best_fitness(final_carry)
    # Creating a jax array block is necessary to force sync
    _ = jax.block_until_ready(jax.numpy.array(fitness_scalar))
    t_exec = time.perf_counter() - t0
    
    return BenchmarkResult(
        framework=framework_name,
        device=adapter.get_device_info(),
        compile_time=t_compile,
        execution_time=t_exec,
        generations_per_sec=num_gens / t_exec,
        best_fitness=fitness_scalar
    )