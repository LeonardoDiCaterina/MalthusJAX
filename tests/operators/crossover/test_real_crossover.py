import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import (
    BinomialCrossover,
    BlendCrossover,
    SimulatedBinaryCrossover,
    UniformCrossover,
)


@pytest.fixture
def crossover_context():
    """Sets up two populations, config, and RNG key for crossover tests."""
    key = jr.PRNGKey(777)
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0), dtype=jnp.float32)
    pop_size = 10

    # Initialize two distinct populations (Parents)
    p1_pop = RealPopulation.init_random(jr.fold_in(key, 0), config, size=pop_size)
    p2_pop = RealPopulation.init_random(jr.fold_in(key, 1), config, size=pop_size)

    return p1_pop, p2_pop, config, key


class TestRealCrossoverHarness:
    """Rigorous validation of Fused 3-Tier Crossovers."""

    @pytest.mark.parametrize(
        "cross_cls, expected_keys",
        [
            (UniformCrossover, 1),
            (BinomialCrossover, 1),
            (BlendCrossover, 2),
            (SimulatedBinaryCrossover, 3),
        ],
    )
    def test_key_budgeting(self, cross_cls, expected_keys):
        """Verifies num_keys_per_atomic_operation matches implementation for ResourceMapper."""
        cross = cross_cls()
        assert cross.num_keys_per_atomic_operation == expected_keys

    def test_uniform_crossover_mixing(self, crossover_context):
        """Validates that UniformCrossover strictly selects values from parents."""
        p1_pop, p2_pop, config, key = crossover_context
        pop_size = len(p1_pop)

        # Lock length for ResourceMapper compatibility
        cross = UniformCrossover(crossover_rate=0.5).set_input_length(pop_size)

        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # In Uniform Crossover, every gene must come from either p1 or p2
        # We check that no "new" values were created (arithmetic purity)
        vals = offspring_pop.genes.values
        from_p1 = vals == p1_pop.genes.values
        from_p2 = vals == p2_pop.genes.values
        assert jnp.all(from_p1 | from_p2)

    def test_blend_crossover_bounds(self, crossover_context):
        """Ensures BLX-alpha generates values within the expanded interval and clips."""
        p1_pop, p2_pop, config, key = crossover_context
        pop_size = len(p1_pop)

        # High crossover rate to ensure recombination occurs
        cross = BlendCrossover(crossover_rate=1.0, alpha=0.5).set_input_length(pop_size)

        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # Verify boundary safety (Tier 1 Clipping)
        vals = offspring_pop.genes.values
        assert jnp.all(vals >= config.bounds[0])
        assert jnp.all(vals <= config.bounds[1])

    '''def test_sbx_statistical_diversity(self, crossover_context):
        """Verifies that SBX creates new values not present in parents (Distributional check)."""
        p1_pop, p2_pop, config, key = crossover_context
        pop_size = len(p1_pop)

        cross = SimulatedBinaryCrossover(crossover_rate=1.0, eta=2.0).set_input_length(pop_size)

        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # SBX should produce values different from both parents
        vals = offspring_pop.genes.values
        assert not jnp.allclose(vals, p1_pop.genes.values)
        assert not jnp.allclose(vals, p2_pop.genes.values)'''

    def test_jit_fusion_reproducibility(self, crossover_context):
        """Ensures the Fused Mode E pass is stable under JIT compilation."""
        p1_pop, p2_pop, config, key = crossover_context
        pop_size = len(p1_pop)

        cross = BinomialCrossover(crossover_rate=0.7).set_input_length(pop_size)
        all_keys = jr.split(key, cross.num_keys((pop_size,)))

        @jax.jit
        def run_cross(k, p1, p2, c):
            return cross(k, p1, p2, c)

        # res_jit and res_raw should be identical
        res_jit = run_cross(all_keys, p1_pop, p2_pop, config)
        res_raw = cross(all_keys, p1_pop, p2_pop, config)

        # Use np.testing for precise array comparison
        np.testing.assert_allclose(res_jit.genes.values, res_raw.genes.values, atol=1e-6)

    def test_fitness_reset_on_spawn(self, crossover_context):
        """Verifies that crossover offspring have fitness reset to NaN."""
        p1_pop, p2_pop, config, key = crossover_context
        pop_size = len(p1_pop)

        # Mock non-NaN fitness for parents
        p1_pop = p1_pop.replace(fitness=jnp.zeros(pop_size))

        cross = UniformCrossover().set_input_length(pop_size)
        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # spawn_offspring contract: fitness must be reset
        assert jnp.all(jnp.isnan(offspring_pop.fitness))
