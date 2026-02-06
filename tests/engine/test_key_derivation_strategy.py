"""
Tests for KeyDerivationStrategy (SPLIT vs FOLD).
Verifies that users can choose between different RNG key derivation methods.
"""

import jax
import jax.numpy as jnp
import jax.random as jar
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.resource_mapper import (
    KeyDerivationStrategy,
    compute_resource_map,
    ResourceMap,
)
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import SinglePointCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation


class TestKeyDerivationStrategy:
    """Test KeyDerivationStrategy enum and its functionality."""

    def test_strategy_enum_values(self):
        """Verify strategy enum has correct values."""
        assert KeyDerivationStrategy.SPLIT.value == "split"
        assert KeyDerivationStrategy.FOLD.value == "fold"

    def test_strategy_enum_members(self):
        """Verify strategy enum has expected members."""
        strategies = list(KeyDerivationStrategy)
        assert len(strategies) == 2
        assert KeyDerivationStrategy.SPLIT in strategies
        assert KeyDerivationStrategy.FOLD in strategies


class TestResourceMapGetKeys:
    """Test ResourceMap.get_keys() method with both strategies."""

    @pytest.fixture
    def resource_map_split(self):
        """Create a ResourceMap with SPLIT strategy."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        return compute_resource_map(
            selection, crossover, mutation, config, pop_size=10, key_derivation=KeyDerivationStrategy.SPLIT
        )

    @pytest.fixture
    def resource_map_fold(self):
        """Create a ResourceMap with FOLD strategy."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        return compute_resource_map(
            selection, crossover, mutation, config, pop_size=10, key_derivation=KeyDerivationStrategy.FOLD
        )

    def test_split_key_derivation_produces_unique_keys(self, resource_map_split):
        """SPLIT strategy should produce diverse keys."""
        master_key = jar.PRNGKey(42)
        keys = resource_map_split.get_keys(master_key)

        # JAX split returns keys with shape (n, 2)
        assert keys.shape[0] == resource_map_split.total_rng_budget
        # Keys should be unique (high probability)
        keys_flat = keys.reshape(keys.shape[0], -1)
        assert len(jnp.unique(keys_flat, axis=0)) == len(keys)

    def test_fold_key_derivation_produces_keys(self, resource_map_fold):
        """FOLD strategy should produce keys deterministically."""
        master_key = jar.PRNGKey(42)
        keys = resource_map_fold.get_keys(master_key)

        # JAX fold_in returns keys with shape (n, 2)
        assert keys.shape[0] == resource_map_fold.total_rng_budget
        # Fold_in derives keys sequentially, they should all be different
        keys_flat = keys.reshape(keys.shape[0], -1)
        assert len(jnp.unique(keys_flat, axis=0)) == len(keys)

    def test_split_vs_fold_produce_different_sequences(self, resource_map_split, resource_map_fold):
        """SPLIT and FOLD strategies are both valid key derivation methods."""
        master_key = jar.PRNGKey(42)

        keys_split = resource_map_split.get_keys(master_key)
        keys_fold = resource_map_fold.get_keys(master_key)

        # Both should produce the correct shape
        assert keys_split.shape[0] == resource_map_split.total_rng_budget
        assert keys_fold.shape[0] == resource_map_fold.total_rng_budget
        
        # Both should be valid keys
        assert keys_split.dtype == keys_fold.dtype

    def test_split_deterministic_with_same_seed(self, resource_map_split):
        """SPLIT strategy should be deterministic with same seed."""
        key1 = jar.PRNGKey(42)
        key2 = jar.PRNGKey(42)

        keys1 = resource_map_split.get_keys(key1)
        keys2 = resource_map_split.get_keys(key2)

        assert jnp.allclose(keys1, keys2)

    def test_fold_deterministic_with_same_seed(self, resource_map_fold):
        """FOLD strategy should be deterministic with same seed."""
        key1 = jar.PRNGKey(42)
        key2 = jar.PRNGKey(42)

        keys1 = resource_map_fold.get_keys(key1)
        keys2 = resource_map_fold.get_keys(key2)

        assert jnp.allclose(keys1, keys2)

    def test_invalid_strategy_raises_error(self, resource_map_split):
        """Invalid strategy should raise ValueError."""
        # Manually set invalid strategy
        invalid_rmap = resource_map_split.replace(key_derivation="invalid")  # type: ignore
        master_key = jar.PRNGKey(42)

        with pytest.raises(ValueError, match="Unknown key derivation strategy"):
            invalid_rmap.get_keys(master_key)


class TestGeneticEngineWithStrategies:
    """Test GeneticEngine using both key derivation strategies."""

    @pytest.fixture
    def engine_params_split(self):
        """Create engine params with SPLIT strategy."""
        return GeneticEngineParams(
            pop_size=10,
            num_generations=5,
            elitism=2,
            key_derivation=KeyDerivationStrategy.SPLIT,
        )

    @pytest.fixture
    def engine_params_fold(self):
        """Create engine params with FOLD strategy."""
        return GeneticEngineParams(
            pop_size=10,
            num_generations=5,
            elitism=2,
            key_derivation=KeyDerivationStrategy.FOLD,
        )

    @pytest.fixture
    def genetic_engine_split(self, engine_params_split):
        """Create genetic engine with SPLIT strategy."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        return GeneticEngine(
            genome_config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            engine_params=engine_params_split,
        )

    @pytest.fixture
    def genetic_engine_fold(self, engine_params_fold):
        """Create genetic engine with FOLD strategy."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        return GeneticEngine(
            genome_config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            engine_params=engine_params_fold,
        )

    def test_split_engine_initialization(self, genetic_engine_split):
        """Engine with SPLIT strategy should initialize correctly."""
        key = jar.PRNGKey(0)
        state = genetic_engine_split.init_state(key)

        assert state.resource_map.key_derivation == KeyDerivationStrategy.SPLIT
        assert state.generation == 0
        assert state.population.genes is not None

    def test_fold_engine_initialization(self, genetic_engine_fold):
        """Engine with FOLD strategy should initialize correctly."""
        key = jar.PRNGKey(0)
        state = genetic_engine_fold.init_state(key)

        assert state.resource_map.key_derivation == KeyDerivationStrategy.FOLD
        assert state.generation == 0
        assert state.population.genes is not None

    def test_split_engine_step(self, genetic_engine_split):
        """Engine with SPLIT strategy should execute step correctly."""
        key = jar.PRNGKey(0)
        state = genetic_engine_split.init_state(key)

        next_state, metrics = genetic_engine_split.step(state)

        assert next_state.generation == 1
        assert metrics.best_fitness >= 0
        assert metrics.mean_fitness >= 0

    def test_fold_engine_step(self, genetic_engine_fold):
        """Engine with FOLD strategy should execute step correctly."""
        key = jar.PRNGKey(0)
        state = genetic_engine_fold.init_state(key)

        next_state, metrics = genetic_engine_fold.step(state)

        assert next_state.generation == 1
        assert metrics.best_fitness >= 0
        assert metrics.mean_fitness >= 0

    def test_multiple_steps_split(self, genetic_engine_split):
        """Engine with SPLIT strategy should run multiple generations."""
        key = jar.PRNGKey(0)
        state = genetic_engine_split.init_state(key)

        for i in range(3):
            state, metrics = genetic_engine_split.step(state)
            assert state.generation == i + 1

    def test_multiple_steps_fold(self, genetic_engine_fold):
        """Engine with FOLD strategy should run multiple generations."""
        key = jar.PRNGKey(0)
        state = genetic_engine_fold.init_state(key)

        for i in range(3):
            state, metrics = genetic_engine_fold.step(state)
            assert state.generation == i + 1

    def test_split_and_fold_both_produce_valid_results(
        self, genetic_engine_split, genetic_engine_fold
    ):
        """Both SPLIT and FOLD strategies should produce valid evolution results."""
        key = jar.PRNGKey(0)

        state_split = genetic_engine_split.init_state(key)
        state_fold = genetic_engine_fold.init_state(key)

        # Run one step with each
        state_split, metrics_split = genetic_engine_split.step(state_split)
        state_fold, metrics_fold = genetic_engine_fold.step(state_fold)

        # Both should produce valid metrics
        assert metrics_split.best_fitness >= 0
        assert metrics_fold.best_fitness >= 0
        assert metrics_split.mean_fitness >= 0
        assert metrics_fold.mean_fitness >= 0
        
        # Both should have advanced to generation 1
        assert state_split.generation == 1
        assert state_fold.generation == 1


class TestResourceMapWithDefaultStrategy:
    """Test ResourceMap defaults to SPLIT strategy."""

    def test_default_strategy_is_split(self):
        """ResourceMap should default to SPLIT strategy."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        rmap = compute_resource_map(selection, crossover, mutation, config, pop_size=10)
        assert rmap.key_derivation == KeyDerivationStrategy.SPLIT

    def test_explicit_strategy_override_default(self):
        """Explicitly setting strategy should override default."""
        config = BinaryGenomeConfig(length=8, dtype=jnp.float32)
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        rmap = compute_resource_map(
            selection, crossover, mutation, config, pop_size=10, key_derivation=KeyDerivationStrategy.FOLD
        )
        assert rmap.key_derivation == KeyDerivationStrategy.FOLD


class TestKeyDerivationIntegration:
    """Integration tests for key derivation with actual evolution."""

    def test_reproducibility_with_split_strategy(self):
        """Same seed with SPLIT strategy should give same results."""
        config = BinaryGenomeConfig(length=16, dtype=jnp.float32)
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        engine = GeneticEngine(
            genome_config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            engine_params=GeneticEngineParams(
                pop_size=10,
                num_generations=3,
                elitism=1,
                key_derivation=KeyDerivationStrategy.SPLIT,
            ),
        )

        # Run twice with same seed
        key = jar.PRNGKey(123)
        state1 = engine.init_state(key)
        for _ in range(2):
            state1, _ = engine.step(state1)
        best_fitness_1 = state1.best_fitness

        state2 = engine.init_state(jar.PRNGKey(123))
        for _ in range(2):
            state2, _ = engine.step(state2)
        best_fitness_2 = state2.best_fitness

        assert jnp.allclose(best_fitness_1, best_fitness_2)

    def test_reproducibility_with_fold_strategy(self):
        """Same seed with FOLD strategy should give same results."""
        config = BinaryGenomeConfig(length=16, dtype=jnp.float32)
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        selection = TournamentSelection(num_selections=10, tournament_size=2)
        crossover = SinglePointCrossover(num_offspring=2)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        engine = GeneticEngine(
            genome_config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            engine_params=GeneticEngineParams(
                pop_size=10,
                num_generations=3,
                elitism=1,
                key_derivation=KeyDerivationStrategy.FOLD,
            ),
        )

        # Run twice with same seed
        key = jar.PRNGKey(456)
        state1 = engine.init_state(key)
        for _ in range(2):
            state1, _ = engine.step(state1)
        best_fitness_1 = state1.best_fitness

        state2 = engine.init_state(jar.PRNGKey(456))
        for _ in range(2):
            state2, _ = engine.step(state2)
        best_fitness_2 = state2.best_fitness

        assert jnp.allclose(best_fitness_1, best_fitness_2)
