import time

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree


# --- Setup Dummy Dataclasses ---
@struct.dataclass
class DummyGenome:
    values: jax.Array


@struct.dataclass
class DummyPopulation:
    genomes: DummyGenome
    fitness: jax.Array


def analyze_kernel(name: str, fn, *args):
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    hlo_lines = len(hlo_text.splitlines())

    out = compiled(*args)
    jax.block_until_ready(out)

    runs = 1000
    start = time.perf_counter()
    for _ in range(runs):
        out = compiled(*args)
    jax.block_until_ready(out)
    end = time.perf_counter()

    avg_ms = ((end - start) / runs) * 1000

    # Check for fusion/tuple/while strings in HLO
    fusions = hlo_text.count("fusion(")
    tuples = hlo_text.count("tuple(")
    whiles = hlo_text.count("while(")

    print(
        f"[{name:<35}] HLO Lines: {hlo_lines:<4} | Time: {avg_ms:.4f} ms | Fusions: {fusions:<2} | Tuples: {tuples:<2} | Whiles: {whiles:<2}"
    )


def run_study():
    pop_size = 195
    num_dims = 9

    flat_values = jnp.ones((pop_size, num_dims))
    flat_fitness = jnp.zeros((pop_size,))

    pytree_pop = DummyPopulation(genomes=DummyGenome(values=flat_values), fitness=flat_fitness)

    print("--- XLA No-Op Discovery Study ---\n")

    # 1. Baseline Flat Math
    def test_baseline_flat(val):
        return val * 0.5 + 0.1

    analyze_kernel("1. Baseline Flat Math", test_baseline_flat, flat_values)

    # 2. PyTree Packing (from flat)
    def test_pack_pytree(val, fit):
        out = test_baseline_flat(val)
        return DummyPopulation(genomes=DummyGenome(values=out), fitness=fit)

    analyze_kernel("2. PyTree Packing (from flat)", test_pack_pytree, flat_values, flat_fitness)

    # 3. PyTree Unpack -> Math -> Pack (Replace)
    def test_pytree_replace(pop):
        out = test_baseline_flat(pop.genomes.values)
        return pop.replace(genomes=pop.genomes.replace(values=out))

    analyze_kernel("3. PyTree .replace()", test_pytree_replace, pytree_pop)

    # 4. Ravel / Unravel
    flat_raveled, unravel_fn = ravel_pytree(pytree_pop)

    def test_ravel_unravel(pop):
        flat, unrav = ravel_pytree(pop)
        # do some math on the flat raveled array
        flat = flat * 0.5 + 0.1
        return unrav(flat)

    analyze_kernel("4. Ravel/Unravel inside JIT", test_ravel_unravel, pytree_pop)

    # 5. Scan over Flat Arrays
    def test_scan_flat(val):
        def body(carry, _):
            return test_baseline_flat(carry), None

        out, _ = jax.lax.scan(body, val, jnp.arange(10))
        return out

    analyze_kernel("5. lax.scan (Flat Array)", test_scan_flat, flat_values)

    # 6. Scan over PyTree with Replace
    def test_scan_pytree(pop):
        def body(carry, _):
            return test_pytree_replace(carry), None

        out, _ = jax.lax.scan(body, pop, jnp.arange(10))
        return out

    analyze_kernel("6. lax.scan (PyTree Replace)", test_scan_pytree, pytree_pop)


if __name__ == "__main__":
    run_study()
