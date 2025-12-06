"""Micro-benchmark harness for operator kernels vs legacy implementations.

Usage:
    python benchmarks/run_benchmarks.py --device cpu --batch 32 --length 128 --iters 5

Outputs JSON results to `benchmarks/results/` (created if missing).
"""
import os
import time
import json
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome


RESULTS_DIR = os.path.join("benchmarks", "results")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def bench_gaussian(batch, length, iters, key_seed=0):
    key = jar.PRNGKey(key_seed)
    config = RealGenomeConfig(length=length, bounds=(-1.0, 1.0))
    op = GaussianMutation(mutation_rate=1.0, mutation_strength=0.05)

    k_init, k_data = jar.split(key)
    data = jar.uniform(k_data, shape=(batch, length), minval=config.bounds[0], maxval=config.bounds[1])

    # Legacy: loop over individuals and call _mutate_one
    leg_key = jar.PRNGKey(key_seed + 1)
    leg_keys = jar.split(leg_key, batch)

    def legacy_run():
        out = []
        for i in range(batch):
            g = RealGenome(values=data[i])
            o = op._mutate_one(leg_keys[i], g, config)
            out.append(o.values)
        return jnp.stack(out)

    # Kernel (vectorized) path
    kern_key = jar.PRNGKey(key_seed + 2)

    # JIT compile wrapper
    jitted_kernel = jax.jit(op.apply_kernel)

    # Warm-up / compile
    t0 = time.perf_counter()
    _ = jitted_kernel(kern_key, data, op)
    compile_time = time.perf_counter() - t0

    # Measure warm runs
    warm_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = jitted_kernel(kern_key, data, op).block_until_ready()
        warm_times.append(time.perf_counter() - t0)

    # Legacy timing (no JIT) - run once for rough baseline
    t0 = time.perf_counter()
    _ = legacy_run()
    legacy_time = time.perf_counter() - t0

    return {
        "operator": "GaussianMutation",
        "batch": batch,
        "length": length,
        "compile_time": compile_time,
        "kernel_warm_mean": float(sum(warm_times) / len(warm_times)),
        "kernel_warm_times": warm_times,
        "legacy_time": legacy_time,
    }


def bench_uniform(batch, length, iters, key_seed=10):
    key = jar.PRNGKey(key_seed)
    config = RealGenomeConfig(length=length, bounds=(-1.0, 1.0))
    op = UniformCrossover(crossover_rate=0.5)

    k1, k2 = jar.split(key)
    p1 = jar.uniform(k1, shape=(batch, length), minval=config.bounds[0], maxval=config.bounds[1])
    p2 = jar.uniform(k2, shape=(batch, length), minval=config.bounds[0], maxval=config.bounds[1])

    # Legacy: loop over pairs
    leg_key = jar.PRNGKey(key_seed + 1)
    leg_keys = jar.split(leg_key, batch)

    def legacy_run():
        out = []
        for i in range(batch):
            g1 = RealGenome(values=p1[i])
            g2 = RealGenome(values=p2[i])
            o = op._cross_one(leg_keys[i], g1, g2, config)
            out.append(o.values)
        return jnp.stack(out)

    kern_key = jar.PRNGKey(key_seed + 2)
    jitted_kernel = jax.jit(op.apply_kernel)

    # Compile
    t0 = time.perf_counter()
    _ = jitted_kernel(kern_key, (p1, p2), op)
    compile_time = time.perf_counter() - t0

    warm_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = jitted_kernel(kern_key, (p1, p2), op).block_until_ready()
        warm_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _ = legacy_run()
    legacy_time = time.perf_counter() - t0

    return {
        "operator": "UniformCrossover",
        "batch": batch,
        "length": length,
        "compile_time": compile_time,
        "kernel_warm_mean": float(sum(warm_times) / len(warm_times)),
        "kernel_warm_times": warm_times,
        "legacy_time": legacy_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    # Set device (via JAX environment) - best-effort
    print(f"Running benchmarks on device={args.device} batch={args.batch} length={args.length}")

    ensure_dir(args.out)

    results = []
    results.append(bench_gaussian(args.batch, args.length, args.iters))
    results.append(bench_uniform(args.batch, args.length, args.iters))

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": args.device,
        "results": results,
    }

    fname = os.path.join(args.out, f"benchmark_{int(time.time())}.json")
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
