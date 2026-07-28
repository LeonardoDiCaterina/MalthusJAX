import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.crossover.binary import SinglePointCrossover, UniformCrossover


@pytest.fixture
def binary_crossover_context():
    """Sets up two binary populations, config, and RNG key for tests."""
    key = jr.PRNGKey(888)
    # Using 12 bits for a clean divisibility check
    config = BinaryGenomeConfig(shape=(12,), p=0.5, dtype=jnp.int32)
    pop_size = 10

    p1_pop = BinaryPopulation.init_random(jr.fold_in(key, 0), config, size=pop_size)
    p2_pop = BinaryPopulation.init_random(jr.fold_in(key, 1), config, size=pop_size)

    return p1_pop, p2_pop, config, key


class TestBinaryCrossoverHarness:
    """Validates Binary 3-Tier Fused Crossovers and ResourceMapper integration."""

    @pytest.mark.parametrize("cross_cls", [UniformCrossover, SinglePointCrossover])
    def test_key_budgeting(self, cross_cls):
        """Verifies that each binary operator requires exactly 1 key for its mask/point."""
        cross = cross_cls()
        assert cross.num_keys_per_atomic_operation == 1

    def test_uniform_crossover_bitwise_integrity(self, binary_crossover_context):
        """Ensures UniformCrossover strictly selects bits from parents (no new values)."""
        p1_pop, p2_pop, config, key = binary_crossover_context
        pop_size = len(p1_pop)

        cross = UniformCrossover(crossover_rate=0.5).set_input_length(pop_size)

        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # Every bit must match either parent 1 or parent 2 at that index
        vals = offspring_pop.genes.values
        from_p1 = vals == p1_pop.genes.values
        from_p2 = vals == p2_pop.genes.values
        assert jnp.all(from_p1 | from_p2)

    def test_single_point_segmentation_robust(self, binary_crossover_context):
        p1_pop, p2_pop, config, key = binary_crossover_context
        pop_size = len(p1_pop)

        # 1. Gimmick: Replace binary values with unique IDs (Indices)
        # This ensures P1 and P2 genes NEVER match.
        p1_unique = jnp.arange(config.shape[0])
        p2_unique = jnp.arange(config.shape[0]) + 1000  # Offset for P2

        # Inject these into the populations
        p1_pop = p1_pop.replace(
            genes=p1_pop.genes.replace(values=jnp.broadcast_to(p1_unique, (pop_size, 12)))
        )
        p2_pop = p2_pop.replace(
            genes=p2_pop.genes.replace(values=jnp.broadcast_to(p2_unique, (pop_size, 12)))
        )

        cross = SinglePointCrossover().set_input_length(pop_size)
        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        vals = offspring_pop.genes.values

        def count_switches(offspring, p1):
            # Now matches_p1 is ONLY true before the crossover point
            matches_p1 = offspring == p1
            return jnp.sum(jnp.abs(jnp.diff(matches_p1.astype(int))))

        # vmap across the population
        switches = jax.vmap(count_switches)(vals, p1_pop.genes.values)

        # Now it will be exactly 1 every time (except if point=0/length,
        # but our logic avoids that)
        assert jnp.all(switches == 1)

    def test_jit_reproducibility(self, binary_crossover_context):
        """Ensures the Fused Mode E pass is bitwise identical under JIT."""
        p1_pop, p2_pop, config, key = binary_crossover_context
        pop_size = len(p1_pop)

        cross = UniformCrossover(crossover_rate=0.5).set_input_length(pop_size)
        all_keys = jr.split(key, cross.num_keys((pop_size,)))

        @jax.jit
        def run_cross(k, p1, p2, c):
            return cross(k, p1, p2, c)

        res_jit = run_cross(all_keys, p1_pop, p2_pop, config)
        res_raw = cross(all_keys, p1_pop, p2_pop, config)

        # Binary integers should be exactly identical
        assert jnp.all(res_jit.genes.values == res_raw.genes.values)

    def test_offspring_spawn_behavior(self, binary_crossover_context):
        """Verifies population integrity: correct size and reset fitness."""
        p1_pop, p2_pop, config, key = binary_crossover_context
        pop_size = len(p1_pop)

        cross = SinglePointCrossover().set_input_length(pop_size)
        all_keys = jr.split(key, cross.num_keys((pop_size,)))
        offspring_pop = cross(all_keys, p1_pop, p2_pop, config)

        # 1. Size Check (Assuming 1 offspring per pair based on current _recombine_one)
        assert len(offspring_pop) == pop_size

        # 2. Fitness Reset
        assert jnp.all(jnp.isnan(offspring_pop.fitness))
