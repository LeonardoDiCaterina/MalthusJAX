import time
import jax
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any
from .adapters import AbstractBenchmarkAdapter

@dataclass
class BenchmarkResult:
    framework: str
    device: str
    pop_size: int
    unroll: int
    compile_time: float
    
    # Statistical Data
    mean_exec_time: float
    std_exec_time: float
    min_exec_time: float
    max_exec_time: float
    mean_gps: float  # Generations Per Second
    
    # Raw Data (Optional, for deep debugging)
    all_times: List[float] = field(default_factory=list)

def run_adapter_benchmark(
    adapter: AbstractBenchmarkAdapter, 
    num_gens: int, 
    seed: int,
    framework_name: str,
    pop_size: int,       # <--- This was missing in your version
    unroll_factor: int = 1, 
    repeats: int = 30    
) -> BenchmarkResult:
    """
    Compiles ONCE, then runs 'repeats' times to get statistical significance.
    Applies 'unroll_factor' to the scan loop.
    """
    master_key = jax.random.PRNGKey(seed)
    
    # 1. Initialization (Get shape/dtypes)
    # We use a dummy key for compilation tracing
    init_carry = adapter.init(master_key)
    
    # 2. Get the Step Function
    step_fn = adapter.make_step_fn()
    
    # 3. Define the Scan Loop with UNROLL
    def scan_loop(carry):
        # This is where the magic happens: unroll=K fuses K steps
        return jax.lax.scan(step_fn, carry, None, length=num_gens, unroll=unroll_factor)
    
    # 4. Compilation (Warmup)
    print(f"[{framework_name}] Compiling (Unroll={unroll_factor})...", end="", flush=True)
    t0 = time.perf_counter()
    
    jit_scan = jax.jit(scan_loop)
    # Force compilation
    compiled_scan = jit_scan.lower(init_carry).compile()
    
    t_compile = time.perf_counter() - t0
    print(f" Done ({t_compile:.4f}s)")
    
    # 5. Statistical Execution Loop
    exec_times = []
    
    # We run 'repeats' times. 
    # For throughput measurement, we can reuse the compiled artifact.
    # We explicitly BLOCK to ensure we measure GPU time, not dispatch time.
    
    for i in range(repeats):
        # Fold key to get unique seed for this run (ensure robustness)
        iter_key = jax.random.fold_in(master_key, i)
        
        # Re-init state (lightweight, just populates buffers)
        iter_carry = adapter.init(iter_key)
        
        # Block before starting timer to clear queue
        jax.block_until_ready(iter_carry)
        
        t_start = time.perf_counter()
        final_carry, _ = compiled_scan(iter_carry)
        
        # Block on result to force synchronization
        jax.block_until_ready(final_carry)
        t_end = time.perf_counter()
        
        exec_times.append(t_end - t_start)

    # 6. Compute Statistics
    exec_times = np.array(exec_times)
    mean_time = float(np.mean(exec_times))
    std_time = float(np.std(exec_times))
    
    if mean_time > 0:
        mean_gps = num_gens / mean_time
    else:
        mean_gps = 0.0

    return BenchmarkResult(
        framework=framework_name,
        device=adapter.get_device_info(),
        pop_size=pop_size,
        unroll=unroll_factor,
        compile_time=t_compile,
        mean_exec_time=mean_time,
        std_exec_time=std_time,
        min_exec_time=float(np.min(exec_times)),
        max_exec_time=float(np.max(exec_times)),
        mean_gps=mean_gps,
        all_times=exec_times.tolist()
    )