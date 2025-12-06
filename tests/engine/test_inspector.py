"""
Tests for the Operator Inspector (Step 2 of Optimization Roadmap).

Tests engine initialization with kernel support detection and mode selection.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.engine.inspector import (
    inspect_operator,
    inspect_engine_operators,
    ExecutionMode,
    OperatorIdentityCard,
    get_kernel_support_summary,
)
from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig


# ==========================================
# Test Fixtures: Custom Operators with Kernel Support
# ==========================================

@struct.dataclass
class MockMutationWithKernel(BaseMutation):
    """Mock mutation operator with full kernel support."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def _mutate_one(self, key, genome, config):
        # Simple mock implementation
        return genome
    
    def num_keys(self, config, input_shape):
        return self.num_offspring * 2  # Custom implementation
    
    def get_output_shape(self, config, input_shape):
        return (self.num_offspring, *input_shape)
    
    def apply_kernel(self, keys, genome, config):
        # Custom kernel implementation
        return self.__call__(keys[0], genome, config)


@struct.dataclass
class MockCrossoverWithKernel(BaseCrossover):
    """Mock crossover operator with full kernel support."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _crossover_pair(self, key, parent1, parent2, config):
        return parent1
    
    def num_keys(self, config, input_shape):
        return 1  # Custom implementation
    
    def get_output_shape(self, config, input_shape):
        return (self.num_offspring, *input_shape)
    
    def apply_kernel(self, keys, parent1, parent2, config):
        # Custom kernel implementation
        return self.__call__(keys[0], parent1, parent2, config)


@struct.dataclass
class MockSelectionWithKernel(BaseSelection):
    """Mock selection operator with full kernel support."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def _select_batch(self, key, fitness_values):
        return jnp.arange(self.num_selections)
    
    def num_keys(self, config, input_shape):
        return 1  # Custom implementation
    
    def get_output_shape(self, config, input_shape):
        return (self.num_selections,)
    
    def apply_kernel(self, keys, fitness_values):
        # Custom kernel implementation
        return self.__call__(keys[0], fitness_values)


@struct.dataclass
class MockMutationLegacy(BaseMutation):
    """Mock mutation operator without kernel support (uses defaults)."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def _mutate_one(self, key, genome, config):
        return genome


@struct.dataclass
class MockCrossoverLegacy(BaseCrossover):
    """Mock crossover operator without kernel support (uses defaults)."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _crossover_pair(self, key, parent1, parent2, config):
        return parent1


@struct.dataclass
class MockSelectionLegacy(BaseSelection):
    """Mock selection operator without kernel support (uses defaults)."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def _select_batch(self, key, fitness_values):
        return jnp.arange(self.num_selections)


# ==========================================
# Test: inspect_operator()
# ==========================================

def test_inspect_mutation_with_kernel():
    """Test inspection of mutation operator with full kernel support."""
    mutation = MockMutationWithKernel(mutation_rate=0.1)
    card = inspect_operator(mutation)
    
    assert isinstance(card, OperatorIdentityCard)
    assert card.operator_type == "mutation"
    assert card.has_num_keys is True
    assert card.has_get_output_shape is True
    assert card.has_apply_kernel is True
    assert card.supports_kernel is True


def test_inspect_crossover_with_kernel():
    """Test inspection of crossover operator with full kernel support."""
    crossover = MockCrossoverWithKernel()
    card = inspect_operator(crossover)
    
    assert card.operator_type == "crossover"
    assert card.supports_kernel is True


def test_inspect_selection_with_kernel():
    """Test inspection of selection operator with full kernel support."""
    selection = MockSelectionWithKernel(num_selections=20)
    card = inspect_operator(selection)
    
    assert card.operator_type == "selection"
    assert card.supports_kernel is True


def test_inspect_mutation_legacy():
    """Test inspection of legacy mutation operator (default implementations)."""
    mutation = MockMutationLegacy(mutation_rate=0.2)
    card = inspect_operator(mutation)
    
    assert card.operator_type == "mutation"
    assert card.has_num_keys is False
    assert card.has_get_output_shape is False
    assert card.has_apply_kernel is False
    assert card.supports_kernel is False


def test_inspect_crossover_legacy():
    """Test inspection of legacy crossover operator."""
    crossover = MockCrossoverLegacy()
    card = inspect_operator(crossover)
    
    assert card.operator_type == "crossover"
    assert card.supports_kernel is False


def test_inspect_selection_legacy():
    """Test inspection of legacy selection operator."""
    selection = MockSelectionLegacy(num_selections=15)
    card = inspect_operator(selection)
    
    assert card.operator_type == "selection"
    assert card.supports_kernel is False


def test_inspect_invalid_operator():
    """Test that inspection fails gracefully for invalid operators."""
    invalid_operator = "not_an_operator"
    
    with pytest.raises(ValueError, match="Unknown operator type"):
        inspect_operator(invalid_operator)


# ==========================================
# Test: inspect_engine_operators()
# ==========================================

def test_inspect_engine_all_kernel_support():
    """Test engine inspection when all operators support kernel interface."""
    mutation = MockMutationWithKernel()
    crossover = MockCrossoverWithKernel()
    selection = MockSelectionWithKernel()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    
    assert result.mode == ExecutionMode.FAST_LANE
    assert result.all_support_kernel is True
    assert result.mutation_card.supports_kernel is True
    assert result.crossover_card.supports_kernel is True
    assert result.selection_card.supports_kernel is True


def test_inspect_engine_all_legacy():
    """Test engine inspection when all operators are legacy."""
    mutation = MockMutationLegacy()
    crossover = MockCrossoverLegacy()
    selection = MockSelectionLegacy()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    
    assert result.mode == ExecutionMode.LEGACY
    assert result.all_support_kernel is False
    assert result.mutation_card.supports_kernel is False
    assert result.crossover_card.supports_kernel is False
    assert result.selection_card.supports_kernel is False


def test_inspect_engine_mixed_support():
    """Test engine inspection with mixed kernel support (should be LEGACY)."""
    # Only mutation has kernel support
    mutation = MockMutationWithKernel()
    crossover = MockCrossoverLegacy()
    selection = MockSelectionLegacy()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    
    assert result.mode == ExecutionMode.LEGACY
    assert result.all_support_kernel is False
    assert result.mutation_card.supports_kernel is True
    assert result.crossover_card.supports_kernel is False
    assert result.selection_card.supports_kernel is False


def test_inspect_engine_two_kernel_one_legacy():
    """Test engine with two kernel operators and one legacy (should be LEGACY)."""
    mutation = MockMutationWithKernel()
    crossover = MockCrossoverWithKernel()
    selection = MockSelectionLegacy()  # Only selection is legacy
    
    result = inspect_engine_operators(mutation, crossover, selection)
    
    assert result.mode == ExecutionMode.LEGACY
    assert result.all_support_kernel is False


# ==========================================
# Test: get_kernel_support_summary()
# ==========================================

def test_kernel_support_summary_fast_lane():
    """Test summary formatting for FAST_LANE mode."""
    mutation = MockMutationWithKernel()
    crossover = MockCrossoverWithKernel()
    selection = MockSelectionWithKernel()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    summary = get_kernel_support_summary(result)
    
    assert "FAST_LANE" in summary
    assert "All operators support kernel: True" in summary
    assert "✓ Full kernel support" in summary


def test_kernel_support_summary_legacy():
    """Test summary formatting for LEGACY mode."""
    mutation = MockMutationLegacy()
    crossover = MockCrossoverLegacy()
    selection = MockSelectionLegacy()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    summary = get_kernel_support_summary(result)
    
    assert "LEGACY" in summary
    assert "All operators support kernel: False" in summary
    assert "✗ Missing:" in summary
    assert "num_keys" in summary
    assert "get_output_shape" in summary
    assert "apply_kernel" in summary


def test_kernel_support_summary_mixed():
    """Test summary formatting with mixed support."""
    mutation = MockMutationWithKernel()
    crossover = MockCrossoverLegacy()
    selection = MockSelectionLegacy()
    
    result = inspect_engine_operators(mutation, crossover, selection)
    summary = get_kernel_support_summary(result)
    
    assert "LEGACY" in summary
    assert "Mutation:  ✓ Full kernel support" in summary
    assert "Crossover: ✗" in summary
    assert "Selection: ✗" in summary


# ==========================================
# Test: Integration with GeneticEngine
# ==========================================

def test_genetic_engine_mode_property():
    """Test that GeneticEngine properly exposes mode property."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    # Create engine with legacy operators
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionLegacy(),
        crossover=MockCrossoverLegacy(),
        mutation=MockMutationLegacy()
    )
    
    # Check mode property works
    assert hasattr(engine, "mode")
    assert engine.mode == ExecutionMode.LEGACY


def test_genetic_engine_mode_fast_lane():
    """Test that GeneticEngine detects FAST_LANE mode."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionWithKernel(),
        crossover=MockCrossoverWithKernel(),
        mutation=MockMutationWithKernel()
    )
    
    assert engine.mode == ExecutionMode.FAST_LANE


def test_genetic_engine_inspection_result_property():
    """Test that GeneticEngine exposes inspection_result property."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionLegacy(),
        crossover=MockCrossoverLegacy(),
        mutation=MockMutationWithKernel()
    )
    
    # Check inspection_result property
    result = engine.inspection_result
    assert result.mode == ExecutionMode.LEGACY
    assert result.mutation_card.supports_kernel is True
    assert result.crossover_card.supports_kernel is False
    assert result.selection_card.supports_kernel is False


def test_genetic_engine_inspection_caching():
    """Test that inspection results are consistent."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionLegacy(),
        crossover=MockCrossoverLegacy(),
        mutation=MockMutationLegacy()
    )
    
    # Multiple accesses should return consistent results
    mode1 = engine.mode
    result1 = engine.inspection_result
    
    mode2 = engine.mode
    result2 = engine.inspection_result
    
    # Results should be consistent (same mode and operator support)
    assert mode1 == mode2
    assert result1.mode == result2.mode
    assert result1.all_support_kernel == result2.all_support_kernel


# ==========================================
# Test: Acceptance Criteria from Roadmap
# ==========================================

def test_acceptance_engine_mode_property_exists():
    """Acceptance: engine.mode property exists and is reliable."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionLegacy(),
        crossover=MockCrossoverLegacy(),
        mutation=MockMutationLegacy()
    )
    
    # Property must exist
    assert hasattr(engine, "mode")
    
    # Property must return ExecutionMode enum
    mode = engine.mode
    assert isinstance(mode, ExecutionMode)
    
    # Property must be reliable (consistent across calls)
    assert engine.mode == mode


def test_acceptance_engine_initializes_legacy_mode():
    """Acceptance: engine initializes in LEGACY mode with existing operators."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    # Use legacy operators (default implementations)
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionLegacy(),
        crossover=MockCrossoverLegacy(),
        mutation=MockMutationLegacy()
    )
    
    # Must be in LEGACY mode
    assert engine.mode == ExecutionMode.LEGACY


def test_acceptance_engine_toggles_to_fast_lane():
    """Acceptance: engine toggles to FAST_LANE after kernel operators added."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    # Use kernel-enabled operators
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionWithKernel(),
        crossover=MockCrossoverWithKernel(),
        mutation=MockMutationWithKernel()
    )
    
    # Must be in FAST_LANE mode
    assert engine.mode == ExecutionMode.FAST_LANE
    assert engine.inspection_result.all_support_kernel is True
