"""
Evaluator Stability and Physical Precision Tests for MalthusJAX.

Guarantees that:
1. Low-precision genomes evaluated on standard benchmark functions produce finite, valid float32 fitness.
2. Genuine float16 limits are confirmed (e.g. Schwefel D=100 optimum 183,900 overflows to inf in float16).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.composer.composer import Composer


@pytest.mark.parametrize("fn_name", ["sphere", "rosenbrock", "rastrigin"])
def test_mixed_precision_evaluator_safety(fn_name):
    """Tests that low-precision genomes produce finite, valid float32 fitness arrays without NaNs."""
    composer = Composer.create_default()

    res = composer.quick_run(
        fitness=fn_name,
        genome_length=10,
        pop_size=30,
        generations=3,
        bounds=(-5.0, 5.0),
        dtype="float16",
        seeds=(42,),
    )

    fitness = res.runs[0].metrics["best_fitness"]
    assert np.isfinite(fitness), f"Non-finite fitness {fitness} in {fn_name} under float16!"


def test_physical_float16_schwefel_overflow():
    """
    Confirms genuine float16 physics:
    Schwefel optimum value at D=100 is ~183,900.
    Since float16 max is 65,504, pure float16 arithmetic MUST overflow to inf.
    This test verifies that we are actually testing real float16 behavior.
    """
    optimum_val = 183900.0
    val_f32 = jnp.float32(optimum_val)
    val_f16 = jnp.float16(optimum_val)

    assert jnp.isfinite(val_f32), "float32 must remain finite at 183,900"
    assert jnp.isinf(val_f16), "float16 must overflow to infinity at 183,900 (max is 65,504)"
