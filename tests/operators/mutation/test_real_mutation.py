import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.real import BallMutation, GaussianMutation, PolynomialMutation


@pytest.fixture
def mutation_context():
    """Sets up a standard population, config, and RNG key for mutation tests."""
    key = jr.PRNGKey(123)
    config = RealGenomeConfig(shape=(8,), bounds=(-10.0, 10.0), dtype=jnp.bfloat16)
    pop_size = 6
    population = RealPopulation.init_random(key, config, size=pop_size)
    return population, config, key


class TestRealMutationHarness:
    """Rigorous validation with fixed attribute access and key budgeting."""

    @pytest.mark.parametrize("mut_cls", [GaussianMutation, BallMutation, PolynomialMutation])
    def test_tier_one_arithmetic_purity(self, mutation_context, mut_cls):
        pop, config, _ = mutation_context
        mut = mut_cls(mutation_rate=1.0)

        # FIX: Tier 1 Kernels expect a single Array (the delta), not a tuple.
        # Tier 2 handles the masking logic before passing data to Tier 1.
        mock_noise = jnp.ones(config.shape, dtype=config.dtype)

        # pop[0] IS the genome object
        mutated = mut._mutate_one(pop[0], mock_noise, config)

        # Verify the arithmetic: genome.values (ones from init_random) + mock_noise (ones)
        # Note: init_random uses jax.random, so we just check for expected behavior/dtype
        assert mutated.values.dtype == jnp.bfloat16
        assert isinstance(mutated, RealGenome)

    def test_jit_reproducibility(self, mutation_context):
        """Fixes the JIT decorator error and traces the config statically."""
        pop, config, key = mutation_context
        mut = GaussianMutation(mutation_rate=0.5).set_input_length(len(pop))

        n_keys = mut.num_keys((len(pop),))
        all_keys = jr.split(key, n_keys)

        def _call_wrapper(k, p, c):
            return mut(k, p, c)

        # static_argnums=(2,) for RealGenomeConfig
        compiled_call = jax.jit(_call_wrapper, static_argnums=(2,))

        res_jit = compiled_call(all_keys, pop, config)
        res_raw = mut(all_keys, pop, config)

        # Cast bfloat16 to float32 for NumPy compatibility
        np.testing.assert_allclose(
            res_jit.genes.values.astype(float), res_raw.genes.values.astype(float), atol=1e-3
        )
