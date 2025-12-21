import time
import jax
import numpy as np
from dataclasses import dataclass, field
from typing import List
from .adapters import AbstractBenchmarkAdapter

@dataclass
class BenchmarkResult:
    framework: str
    device: str
    pop_size: int
    unroll: int
    compile_time: float
    mean_exec_time: float
    std_exec_time: float
    mean_gps: float
    
    # New Field
    best_fitness_final: float 
    
    all_times: List[float] = field(default_factory=list)

def run_adapter_benchmark(
    adapter: AbstractBenchmarkAdapter, 
    num_gens: int, 
    seed: int,
    framework_name: str,
    pop_size: int,
    unroll_factor: int = 1, 
    repeats: int = 30    
) -> BenchmarkResult:
    
    master_key = jax.random.PRNGKey(seed)
    init_carry = adapter.init(master_key)
    step_fn = adapter.make_step_fn()
    
    def scan_loop(carry):
        return jax.lax.scan(step_fn, carry, None, length=num_gens, unroll=unroll_factor)
    
    print(f"[{framework_name}] Compiling (Unroll={unroll_factor})...", end="", flush=True)
    t0 = time.perf_counter()
    jit_scan = jax.jit(scan_loop)
    compiled_scan = jit_scan.lower(init_carry).compile()
    t_compile = time.perf_counter() - t0
    print(f" Done ({t_compile:.4f}s)")
    
    exec_times = []
    final_fitnesses = []

    for i in range(repeats):
        iter_key = jax.random.fold_in(master_key, i)
        iter_carry = adapter.init(iter_key)
        jax.block_until_ready(iter_carry)
        
        t_start = time.perf_counter()
        final_carry, _ = compiled_scan(iter_carry)
        jax.block_until_ready(final_carry)
        t_end = time.perf_counter()
        
        exec_times.append(t_end - t_start)
        
        # Extract fitness from FINAL state (Zero overhead inside loop)
        fit = adapter.extract_best_fitness(final_carry)
        final_fitnesses.append(fit)

    exec_times = np.array(exec_times)
    mean_time = float(np.mean(exec_times))
    mean_gps = num_gens / mean_time if mean_time > 0 else 0.0
    
    # We report the average final fitness across the 30 runs (robust to outliers)
    best_final = float(np.average(final_fitnesses))

    return BenchmarkResult(
        framework=framework_name,
        device=adapter.get_device_info(),
        pop_size=pop_size,
        unroll=unroll_factor,
        compile_time=t_compile,
        mean_exec_time=mean_time,
        std_exec_time=float(np.std(exec_times)),
        mean_gps=mean_gps,
        best_fitness_final=best_final,
        all_times=exec_times.tolist()
    )