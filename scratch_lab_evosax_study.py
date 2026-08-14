import time

import jax
import jax.numpy as jnp


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
    print(f"[{name}] HLO Lines: {hlo_lines:<5} | Time: {avg_ms:.4f} ms")


def evosax_study():
    pop_size = 195
    num_dims = 9

    rng = jax.random.PRNGKey(0)

    # Simulate the flat array state of EvoSAX
    population = jnp.zeros((pop_size, num_dims))
    fitness = jnp.zeros((pop_size,))

    crossover_rate = 0.5
    mutation_std = 0.1
    elite_ratio = 0.5
    num_elites = int(pop_size * elite_ratio)

    def evosax_ask_kernel(rng, population, fitness):
        k1, k2, k_cross, k_mut = jax.random.split(rng, 4)

        # 1. Selection (Truncation)
        idx = jnp.argsort(fitness)
        sorted_pop = population[idx]

        # 2. Parent sampling
        p = jnp.arange(pop_size) < num_elites
        parents_1 = jax.random.choice(k1, sorted_pop, (pop_size,), p=p)
        parents_2 = jax.random.choice(k2, sorted_pop, (pop_size,), p=p)

        # 3. Fused Crossover + Mutation via vmap
        def crossover_mutate(rng_c, rng_m, p1, p2):
            mask = jax.random.uniform(rng_c, p1.shape) < crossover_rate
            offspring = jnp.where(mask, p2, p1)
            noise = jax.random.normal(rng_m, offspring.shape) * mutation_std
            return offspring + noise

        return jax.vmap(crossover_mutate)(
            jax.random.split(k_cross, pop_size),
            jax.random.split(k_mut, pop_size),
            parents_1,
            parents_2,
        )

    print("--- In-Depth Study: Flat Native Kernel (EvoSAX equivalent) ---")
    analyze_kernel("Flat Native ask()", evosax_ask_kernel, rng, population, fitness)


if __name__ == "__main__":
    evosax_study()
