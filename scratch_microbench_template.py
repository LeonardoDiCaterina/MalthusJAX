import time

import jax
import jax.numpy as jnp


def analyze_kernel(name: str, fn, *args):
    """Compiles a JAX function, counts HLO lines, and benchmarks speed."""
    # 1. Extract HLO
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    hlo_lines = len(hlo_text.splitlines())

    # 2. Warmup
    out = compiled(*args)
    jax.block_until_ready(out)

    # 3. Benchmark
    runs = 1000
    start = time.perf_counter()
    for _ in range(runs):
        out = compiled(*args)
    jax.block_until_ready(out)
    end = time.perf_counter()

    avg_ms = ((end - start) / runs) * 1000
    print(f"[{name}] HLO Lines: {hlo_lines:<5} | Time: {avg_ms:.4f} ms")
    return hlo_lines, avg_ms


def run_microbenchmark():
    key = jax.random.PRNGKey(0)
    pop = jnp.ones((195, 9))

    # --- Hypothesis Test: Double vmap vs Fused vmap ---

    def baseline_double_vmap(key, pop):
        """Standard approach: crossover then mutate in separate passes."""

        def crossover(x):
            return x * 0.5

        def mutate(x):
            return x + 0.1

        pop2 = jax.vmap(crossover)(pop)
        return jax.vmap(mutate)(pop2)

    def candidate_fused_vmap(key, pop):
        """Candidate approach: fused operations in one pass."""

        def fused_op(x):
            return (x * 0.5) + 0.1

        return jax.vmap(fused_op)(pop)

    print("Running Microbenchmark...\n")
    analyze_kernel("Baseline (Double)", baseline_double_vmap, key, pop)
    analyze_kernel("Candidate (Fused)", candidate_fused_vmap, key, pop)


if __name__ == "__main__":
    run_microbenchmark()
