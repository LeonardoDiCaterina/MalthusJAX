"""Parity test for EvoSaxMimicSelection against EvoSAX SimpleGA selection logic.

Validates that EvoSaxMimicSelection produces identical parent and elite indices to
upstream EvoSAX SimpleGA uniform-over-elites selection across all key representations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.operators.selection.evosax_mimic import EvoSaxMimicSelection


def test_evosax_mimic_selection_num_keys():
    sel = EvoSaxMimicSelection(num_selections=16, elite_k=4)
    assert sel.num_keys_per_atomic_operation == 1


@pytest.mark.parametrize(
    "key_shape_mode",
    ["scalar", "1d_array", "batch_array"],
)
def test_evosax_mimic_selection_matches_evosax_logic(key_shape_mode: str):
    base_key = jax.random.PRNGKey(123)
    k_fit, k_sel = jax.random.split(base_key)

    pop_size = 20
    elite_k = 5
    num_selections = 16
    n_elites = 2

    fitness = jax.random.normal(k_fit, shape=(pop_size,))

    if key_shape_mode == "scalar":
        key_input = k_sel
        expected_rng = k_sel
    elif key_shape_mode == "1d_array":
        key_input = jnp.expand_dims(k_sel, 0)
        expected_rng = k_sel
    else:
        key_input = jax.random.split(k_sel, 3)
        expected_rng = key_input[0]

    selector = EvoSaxMimicSelection(num_selections=num_selections, elite_k=elite_k).set_n_elites(
        n_elites
    )

    # 1. Direct call with raw fitness array
    parent_idx_raw, elite_idx_raw = selector(key_input, fitness)

    # 2. Direct call with BasePopulation PyTree
    pop = BasePopulation(
        genes=RealGenome(values=jnp.zeros((pop_size, 2))),
        fitness=fitness,
    )
    parent_idx_pop, elite_idx_pop = selector(key_input, pop)

    # 3. Expected selection using upstream EvoSAX SimpleGA logic
    idx = jnp.argsort(fitness)
    p = (jnp.arange(pop_size) < elite_k).astype(jnp.float32)
    p_norm = p / jnp.sum(p)
    expected_parents = jax.random.choice(
        expected_rng, idx, shape=(num_selections,), replace=True, p=p_norm
    )
    expected_elites = idx[:n_elites]

    # Exact parity assertion
    np.testing.assert_array_equal(np.asarray(parent_idx_raw), np.asarray(expected_parents))
    np.testing.assert_array_equal(np.asarray(elite_idx_raw), np.asarray(expected_elites))
    np.testing.assert_array_equal(np.asarray(parent_idx_pop), np.asarray(expected_parents))
    np.testing.assert_array_equal(np.asarray(elite_idx_pop), np.asarray(expected_elites))

    # All selected parents must belong to the top elite_k
    top_k_indices = set(np.asarray(idx[:elite_k]))
    for parent in np.asarray(parent_idx_raw):
        assert int(parent) in top_k_indices
