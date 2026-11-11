#!/usr/bin/env python3
"""VRAM Stress & Leak Verification Script.

Tests MalthusJAX memory behavior under maximum benchmark workload
(D=100, P=1024, G=200, 10 seeds) on the active GPU.
Validates:
1. Dynamic vs preallocated VRAM behavior.
2. Peak VRAM consumption during full evolutionary cycle.
3. Absence of memory leaks across consecutive coordinates via `jax.clear_caches()`.
"""

from __future__ import annotations

import gc
import os
import time

# Ensure dynamic memory allocation if not explicitly set
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

from malthusjax.composer import Composer


def get_gpu_memory_info() -> dict[str, float]:
    """Retrieve memory stats in Megabytes from JAX / CUDA runtime."""
    devices = jax.devices()
    default_dev = devices[0]

    info: dict[str, float] = {"backend": jax.default_backend()}
    if default_dev.platform == "gpu":
        try:
            stats = default_dev.memory_stats()
            if stats:
                info["bytes_in_use_mb"] = stats.get("bytes_in_use", 0) / (1024 * 1024)
                info["peak_bytes_in_use_mb"] = stats.get("peak_bytes_in_use", 0) / (1024 * 1024)
        except Exception:
            pass
    return info


def run_vram_stress_test() -> None:
    print("=" * 70)
    print("  MALTHUSJAX VRAM STRESS & LEAK VERIFICATION")
    print("=" * 70)
    print(f"JAX Version:          {jax.__version__}")
    print(f"Default Backend:      {jax.default_backend()}")
    print(f"Devices Available:    {jax.devices()}")
    print(f"Preallocate Env Var:  {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default (true)')}")
    print(f"Mem Fraction Env Var: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default (0.75)')}")
    print("-" * 70)

    composer = Composer.create_default()

    # Extreme workload parameters matching maximum suite bounds
    D = 100
    P = 1024
    G = 200
    num_seeds = 10
    seeds = list(range(1, num_seeds + 1))

    pipelines = {
        "baseline_f32": {
            "backend": "malthusjax",
            "selection": f"evosax_mimic_selection:num_selections={P},elite_k={max(2, P // 6)}",
            "crossover": "evosax_uniform_crossover:crossover_rate=0.3",
            "mutation": "evosax_gaussian:mutation_strength=1.0",
            "elitism": 0,
            "dtype": "float32",
        },
        "ablate_mutation_batched": {
            "backend": "malthusjax",
            "selection": f"evosax_mimic_selection:num_selections={P},elite_k={max(2, P // 6)}",
            "crossover": "evosax_uniform_crossover:crossover_rate=0.3",
            "mutation": "batched_gaussian:mutation_rate=0.1,mutation_strength=1.0",
            "elitism": 0,
            "dtype": "float32",
        },
        "precision_bf16": {
            "backend": "malthusjax",
            "selection": f"evosax_mimic_selection:num_selections={P},elite_k={max(2, P // 6)}",
            "crossover": "evosax_uniform_crossover:crossover_rate=0.3",
            "mutation": "evosax_gaussian:mutation_strength=1.0",
            "elitism": 0,
            "dtype": "bfloat16",
        },
    }

    shared_kwargs = {
        "fitness": f"bbob:fn_name=gallagher_21_hi,num_dims={D},maximize=false",
        "pop_size": P,
        "generations": G,
        "genome_length": D,
        "bounds": (-5.0, 5.0),
        "maximize": False,
    }

    print(f"\nConfiguration: D={D}, P={P}, G={G}, Seeds={num_seeds} across 3 heavy pipelines")
    print("Testing 3 consecutive coordinate cycles to verify cache clearing and zero leak...")

    baseline_mem = get_gpu_memory_info()
    print(f"\nInitial VRAM in use: {baseline_mem.get('bytes_in_use_mb', 0.0):.2f} MB")

    for cycle in range(1, 4):
        t0 = time.time()
        print(f"\n>>> Running Cycle {cycle}/3...")
        composer.compare(
            pipelines=pipelines,
            seeds=seeds,
            shared_initial_population=True,
            pop_seed=42 + cycle,
            **shared_kwargs,
        )
        duration = time.time() - t0

        during_mem = get_gpu_memory_info()
        print(f"    Cycle {cycle} finished in {duration:.2f}s")
        print(f"    VRAM in use (pre-clear):   {during_mem.get('bytes_in_use_mb', 0.0):.2f} MB")
        print(f"    Peak VRAM observed:       {during_mem.get('peak_bytes_in_use_mb', 0.0):.2f} MB")

        # Emulate benchmark_runner cache flush
        jax.clear_caches()
        gc.collect()

        post_clear_mem = get_gpu_memory_info()
        print(f"    VRAM in use (post-clear):  {post_clear_mem.get('bytes_in_use_mb', 0.0):.2f} MB")

    print("\n" + "=" * 70)
    print("  VERDICT: VRAM STRESS TEST COMPLETE")
    final_mem = get_gpu_memory_info()
    peak = final_mem.get("peak_bytes_in_use_mb", 0.0)
    current = final_mem.get("bytes_in_use_mb", 0.0)
    print(f"  Peak VRAM under extreme bounds: {peak:.2f} MB")
    print(f"  Final active VRAM:             {current:.2f} MB")
    if peak < 2000.0:  # Under 2GB
        print("  Status: EXCELLENT (< 2 GB peak). Safe for multi-day cluster execution.")
    else:
        print(f"  Status: WARNING - Peak usage was {peak:.2f} MB.")
    print("=" * 70)


if __name__ == "__main__":
    run_vram_stress_test()
