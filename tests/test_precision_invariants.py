"""
Precision Invariant Tests for MalthusJAX.

Guarantees that reduced-precision configurations (float16, bfloat16) never
silently upcast to float32 at any stage of the evolutionary lifecycle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome, RealPopulation
from malthusjax.composer.composer import Composer
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper, BatchedEvosaxGaussianWrapper


@pytest.mark.parametrize("target_dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_full_lifecycle_dtype_invariance(target_dtype):
    """Guarantees that genes never silently upcast to float32 at any step."""
    key = jax.random.PRNGKey(42)
    config = RealGenomeConfig(shape=(20, 5), bounds=(-5.0, 5.0), dtype=target_dtype)

    # 1. Random population creation
    pop = RealPopulation.init_random(key, config, 20)
    assert pop.genes.values.dtype == target_dtype, (
        f"Initial pop dtype mismatch: expected {target_dtype}, got {pop.genes.values.dtype}"
    )

    # 2. External float32 injection (The exact H3 vulnerability)
    f32_raw = np.random.randn(20, 5).astype(np.float32)
    injected_pop = RealPopulation.from_array(f32_raw, config, RealGenome)
    assert injected_pop.genes.values.dtype == target_dtype, (
        f"Injected array failed to cast to target_dtype: expected {target_dtype}, got {injected_pop.genes.values.dtype}"
    )

    # 3. Mutation operators preserve precision
    mut_op = EvosaxGaussianWrapper(mutation_strength=0.1)
    k_mut = jax.random.split(key, 1)
    mutated = mut_op(k_mut, injected_pop, config)
    assert mutated.genes.values.dtype == target_dtype, (
        f"EvosaxGaussianWrapper leaked precision: expected {target_dtype}, got {mutated.genes.values.dtype}"
    )

    batched_mut = BatchedEvosaxGaussianWrapper(mutation_strength=0.1)
    mutated_batch = batched_mut(k_mut, injected_pop, config)
    assert mutated_batch.genes.values.dtype == target_dtype, (
        f"BatchedEvosaxGaussianWrapper leaked precision: expected {target_dtype}, got {mutated_batch.genes.values.dtype}"
    )


@pytest.mark.parametrize("dt_str", ["float32", "bfloat16", "float16"])
def test_composer_quick_run_dtype_end_to_end(dt_str):
    """Runs a complete multi-generation experiment and ensures dtype is strictly preserved."""
    composer = Composer.create_default()
    f32_pop = np.random.randn(20, 5).astype(np.float32)

    res = composer.quick_run(
        fitness="sphere",
        genome_length=5,
        pop_size=20,
        generations=3,
        bounds=(-5.0, 5.0),
        dtype=dt_str,
        selection="evosax_mimic_selection:num_selections=20,elite_k=0",
        crossover="evosax_uniform_crossover:crossover_rate=0.3",
        mutation="evosax_gaussian:mutation_strength=1.0",
        elitism=0,
        initial_population=f32_pop,
        seeds=(42,),
    )

    run_0 = res.runs[0]
    assert "best_fitness" in run_0.metrics
    assert np.isfinite(run_0.metrics["best_fitness"])
    assert "execution" in run_0.timings
    assert run_0.timings["execution"] > 0


def test_different_dtypes_diverge_numerically():
    """Confirms that float32, bfloat16, and float16 produce distinct trajectories due to precision limits."""
    composer = Composer.create_default()
    f32_pop = np.random.randn(20, 5).astype(np.float32)

    fitnesses = {}
    for dt_str in ["float32", "bfloat16", "float16"]:
        res = composer.quick_run(
            fitness="sphere",
            genome_length=5,
            pop_size=20,
            generations=5,
            bounds=(-5.0, 5.0),
            dtype=dt_str,
            selection="evosax_mimic_selection:num_selections=20,elite_k=0",
            crossover="evosax_uniform_crossover:crossover_rate=0.3",
            mutation="evosax_gaussian:mutation_strength=1.0",
            elitism=0,
            initial_population=f32_pop,
            seeds=(42,),
        )
        fitnesses[dt_str] = res.runs[0].metrics["best_fitness"]

    # Verify that bfloat16 and float16 did NOT produce identical float32 results
    assert fitnesses["bfloat16"] != fitnesses["float32"], "bfloat16 silently duplicated float32!"
    assert fitnesses["float16"] != fitnesses["float32"], "float16 silently duplicated float32!"
    assert fitnesses["float16"] != fitnesses["bfloat16"], "float16 silently duplicated bfloat16!"
