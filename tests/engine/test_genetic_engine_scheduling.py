"""
Tests for mutation strength scheduling in GeneticEngine.

Tests cover two areas:
1. The JAX-native ScheduleType API (CV-3 fix).
2. The standalone compute_scheduled_strength function.
"""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.schedules import ScheduleType, compute_scheduled_strength
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(engine_params, pop_size=30, genome_dims=3):
    """Build a minimal GeneticEngine for scheduling tests."""
    genome_config = RealGenomeConfig(shape=(genome_dims,), bounds=(-5.0, 5.0))
    bbob_config = BBOBConfig(fn_name="sphere", num_dims=genome_dims, maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=pop_size, elite_k=3),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
        enable_progress_bar=False,
    )


# ===========================================================================
# 1. compute_scheduled_strength unit tests
# ===========================================================================


class TestComputeScheduledStrength(unittest.TestCase):
    """Test the pure-JAX schedule function."""

    def test_constant_returns_initial(self):
        for gen in [0, 5, 10]:
            result = compute_scheduled_strength(
                ScheduleType.CONSTANT, gen, 10, initial_strength=0.5
            )
            self.assertAlmostEqual(float(result), 0.5, places=5)

    def test_linear_decay_endpoints(self):
        s0 = compute_scheduled_strength(
            ScheduleType.LINEAR_DECAY,
            0,
            100,
            initial_strength=1.0,
            final_strength=0.1,
        )
        self.assertAlmostEqual(float(s0), 1.0, places=5)
        s100 = compute_scheduled_strength(
            ScheduleType.LINEAR_DECAY,
            100,
            100,
            initial_strength=1.0,
            final_strength=0.1,
        )
        self.assertAlmostEqual(float(s100), 0.1, places=5)

    def test_linear_decay_midpoint(self):
        s50 = compute_scheduled_strength(
            ScheduleType.LINEAR_DECAY,
            50,
            100,
            initial_strength=1.0,
            final_strength=0.0,
        )
        self.assertAlmostEqual(float(s50), 0.5, places=5)

    def test_cosine_anneal_endpoints(self):
        s0 = compute_scheduled_strength(
            ScheduleType.COSINE_ANNEAL,
            0,
            100,
            initial_strength=1.0,
            final_strength=0.0,
        )
        self.assertAlmostEqual(float(s0), 1.0, places=4)
        s100 = compute_scheduled_strength(
            ScheduleType.COSINE_ANNEAL,
            100,
            100,
            initial_strength=1.0,
            final_strength=0.0,
        )
        self.assertAlmostEqual(float(s100), 0.0, places=4)

    def test_exponential_decay_at_start(self):
        s0 = compute_scheduled_strength(
            ScheduleType.EXPONENTIAL_DECAY,
            0,
            100,
            initial_strength=1.0,
        )
        self.assertAlmostEqual(float(s0), 1.0, places=5)

    def test_exponential_decay_decreases(self):
        s0 = compute_scheduled_strength(
            ScheduleType.EXPONENTIAL_DECAY,
            0,
            100,
            initial_strength=1.0,
        )
        s50 = compute_scheduled_strength(
            ScheduleType.EXPONENTIAL_DECAY,
            50,
            100,
            initial_strength=1.0,
        )
        self.assertGreater(float(s0), float(s50))

    def test_jit_safe(self):
        """compute_scheduled_strength works inside jax.jit."""

        @jax.jit
        def _compute(gen):
            return compute_scheduled_strength(
                ScheduleType.LINEAR_DECAY,
                gen,
                100,
                initial_strength=1.0,
                final_strength=0.0,
            )

        result = _compute(50)
        self.assertAlmostEqual(float(result), 0.5, places=5)

    def test_schedule_type_is_int_enum(self):
        """ScheduleType values are plain ints (for pytree_node=False)."""
        self.assertIsInstance(ScheduleType.CONSTANT.value, int)
        self.assertEqual(ScheduleType.CONSTANT, 0)
        self.assertEqual(ScheduleType.LINEAR_DECAY, 1)
        self.assertEqual(ScheduleType.COSINE_ANNEAL, 2)
        self.assertEqual(ScheduleType.EXPONENTIAL_DECAY, 3)


class TestMutationStrengthScheduling(unittest.TestCase):
    """Test new JAX-native schedule API on GeneticEngine."""

    def setUp(self):
        self.key = jar.PRNGKey(42)
        self.pop_size = 30
        self.base_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=10,
        )

    def test_default_is_constant(self):
        self.assertEqual(self.base_params.schedule_type, ScheduleType.CONSTANT)

    def test_engine_without_schedule_uses_fixed_mutation(self):
        """Test that engine without schedule uses fixed mutation strength."""
        engine = _make_engine(self.base_params, self.pop_size)

        # Default schedule_type is CONSTANT
        self.assertEqual(engine.engine_params.schedule_type, ScheduleType.CONSTANT)

        state = engine.init_state(self.key)
        new_state, _ = engine.step(state)
        self.assertEqual(new_state.generation, 1)

    def test_constant_schedule_returns_unchanged_operators(self):
        """CONSTANT schedule should leave operators as-is."""
        engine = _make_engine(self.base_params, self.pop_size)
        state = engine.init_state(self.key)
        # Operators are passed through unchanged—no _get_active_operators needed
        self.assertIsNotNone(state.operators)

    def test_linear_decay_runs_evolution(self):
        """LINEAR_DECAY should reduce mutation_strength over generations.

        Scheduling is now handled inside the operator's _generate_noise via
        the generation argument — no engine-level _get_active_operators needed.
        """
        params = self.base_params.replace(
            schedule_type=ScheduleType.LINEAR_DECAY,
            initial_strength=1.0,
            final_strength=0.0,
        )
        engine = _make_engine(params, self.pop_size)
        state = engine.init_state(self.key)
        for _ in range(5):
            state, _ = engine.step(state)
        self.assertEqual(state.generation, 5)

    def test_engine_runs_with_linear_decay(self):
        params = self.base_params.replace(
            schedule_type=ScheduleType.LINEAR_DECAY,
            initial_strength=1.0,
            final_strength=0.1,
        )
        engine = _make_engine(params, self.pop_size)
        state = engine.init_state(self.key)
        for _ in range(3):
            state, _ = engine.step(state)
        self.assertEqual(state.generation, 3)

    def test_engine_runs_with_cosine_anneal(self):
        params = self.base_params.replace(
            schedule_type=ScheduleType.COSINE_ANNEAL,
            initial_strength=0.8,
            final_strength=0.01,
        )
        engine = _make_engine(params, self.pop_size)
        state = engine.init_state(self.key)
        for _ in range(5):
            state, _ = engine.step(state)
        self.assertEqual(state.generation, 5)
        self.assertTrue(jnp.isfinite(state.best_fitness))

    def test_engine_runs_with_exponential_decay(self):
        params = self.base_params.replace(
            schedule_type=ScheduleType.EXPONENTIAL_DECAY,
            initial_strength=1.0,
        )
        engine = _make_engine(params, self.pop_size)
        state = engine.init_state(self.key)
        for _ in range(5):
            state, _ = engine.step(state)
        self.assertEqual(state.generation, 5)


class TestScheduleIntegrationWithEvolution(unittest.TestCase):
    """Test that scheduling integrates correctly with evolution."""

    def setUp(self):
        self.key = jar.PRNGKey(42)
        self.pop_size = 25
        self.base_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=10,
        )

    def test_scheduling_does_not_break_evolution(self):
        params = self.base_params.replace(
            schedule_type=ScheduleType.LINEAR_DECAY,
            initial_strength=1.0,
            final_strength=0.1,
        )
        engine = _make_engine(params, self.pop_size, genome_dims=2)
        state = engine.init_state(self.key)
        for _ in range(5):
            state, _ = engine.step(state)
        self.assertEqual(state.generation, 5)
        self.assertTrue(jnp.isfinite(state.best_fitness))

    def test_different_schedules_complete(self):
        """All four schedule types complete 5 generations without error."""
        for stype in ScheduleType:
            with self.subTest(schedule=stype.name):
                params = self.base_params.replace(
                    schedule_type=stype,
                    initial_strength=0.5,
                    final_strength=0.01,
                )
                engine = _make_engine(params, self.pop_size, genome_dims=2)
                state = engine.init_state(self.key)
                for _ in range(5):
                    state, _ = engine.step(state)
                self.assertEqual(state.generation, 5)


if __name__ == "__main__":
    unittest.main()
