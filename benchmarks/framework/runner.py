import time
import jax
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
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
    
    # Fitness fields
    best_fitness_final: float
    fitness_std: float  # Standard deviation of final fitness across repeats
    
    all_times: List[float] = field(default_factory=list)

def extract_hlo_from_adapter(
    adapter: AbstractBenchmarkAdapter,
    num_gens: int,
    seed: int,
    framework_name: str,
    unroll_factor: int = 1,
    output_path: Optional[str] = None
) -> str:
    """
    Compile the adapter's evolution loop and extract HLO representation.
    
    Args:
        adapter: The benchmark adapter to compile
        num_gens: Number of generations for the evolution loop
        seed: Random seed for initialization
        framework_name: Name of the framework (for logging)
        unroll_factor: Loop unrolling factor
        output_path: Optional path to save HLO text file
        
    Returns:
        HLO text representation as a string
    """
    master_key = jax.random.PRNGKey(seed)
    init_carry = adapter.init(master_key)
    step_fn = adapter.make_step_fn()
    
    def scan_loop(carry):
        return jax.lax.scan(step_fn, carry, None, length=num_gens, unroll=unroll_factor)
    
    print(f"[{framework_name}] Compiling (Unroll={unroll_factor}) to extract HLO...", end="", flush=True)
    t0 = time.perf_counter()
    jit_scan = jax.jit(scan_loop)
    compiled_scan = jit_scan.lower(init_carry).compile()
    t_compile = time.perf_counter() - t0
    print(f" Done ({t_compile:.4f}s)")
    
    # Extract HLO text representation
    hlo_text = compiled_scan.as_text()
    
    # Optionally save to file
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(hlo_text)
        print(f"HLO saved to: {output_path}")
    
    return hlo_text

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
    final_fitnesses = np.array(final_fitnesses)
    mean_time = float(np.mean(exec_times))
    mean_gps = num_gens / mean_time if mean_time > 0 else 0.0
    
    # We report the average final fitness across the 30 runs (robust to outliers)
    best_final = float(np.mean(final_fitnesses))
    fitness_std = float(np.std(final_fitnesses))

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
        fitness_std=fitness_std,
        all_times=exec_times.tolist()
    )