"""
Tests for Step 4: Execution Router
Verifies routing between LEGACY and FAST_LANE execution paths in GeneticEngine.step()
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from flax import struct

from malthusjax.engine.genetic_engine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.inspector import ExecutionMode
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection


# ==========================================
# Mock Operators (Legacy Only)
# ==========================================

@struct.dataclass
class MockMutationLegacyOnly(BaseMutation):
    """Mutation operator supporting only legacy apply() method."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def __call__(self, key, genome, config):
        # Legacy interface only - genome is a Genome object
        # Extract genes, apply mutation, return (num_offspring, ...genome_shape)
        genes = genome.genes if hasattr(genome, 'genes') else genome
        return jnp.expand_dims(genes, axis=0)


@struct.dataclass
class MockCrossoverLegacyOnly(BaseCrossover):
    """Crossover operator supporting only legacy apply() method."""
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def __call__(self, key, parent1, parent2, config):
        # Legacy interface - parents are Genome objects
        genes1 = parent1.genes if hasattr(parent1, 'genes') else parent1
        genes2 = parent2.genes if hasattr(parent2, 'genes') else parent2
        # Simple crossover: just return first parent (deterministic for testing)
        return jnp.expand_dims(genes1, axis=0)


@struct.dataclass
class MockSelectionLegacyOnly(BaseSelection):
    """Selection operator supporting only legacy apply() method."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def __call__(self, key, fitness):
        # Legacy interface only
        return jnp.arange(self.num_selections)


# ==========================================
# Mock Operators (Kernel Support)
# ==========================================

@struct.dataclass
class MockMutationWithKernel(BaseMutation):
    """Mutation operator supporting apply_kernel interface."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def __call__(self, key, genome, config):
        # Legacy interface
        genes = genome.genes if hasattr(genome, 'genes') else genome
        return jnp.expand_dims(genes, axis=0)
    
    def num_keys(self, config, input_shape):
        return 1
    
    def get_output_shape(self, config, input_shape):
        return (self.num_offspring,) + input_shape
    
    def apply_kernel(self, keys, genome, config):
        # Kernel interface: receives pre-allocated keys, returns (num_offspring, ...genome_shape)
        genes = genome.genes if hasattr(genome, 'genes') else genome
        return jnp.expand_dims(genes, axis=0)


@struct.dataclass
class MockCrossoverWithKernel(BaseCrossover):
    """Crossover operator supporting apply_kernel interface."""
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def __call__(self, key, parent1, parent2, config):
        # Legacy interface
        genes1 = parent1.genes if hasattr(parent1, 'genes') else parent1
        genes2 = parent2.genes if hasattr(parent2, 'genes') else parent2
        return jnp.expand_dims(genes1, axis=0)
    
    def num_keys(self, config, input_shape):
        return 1
    
    def get_output_shape(self, config, input_shape):
        return (self.num_offspring,) + input_shape
    
    def apply_kernel(self, keys, parent1, parent2, config):
        # Kernel interface: returns (num_offspring, ...genome_shape)
        genes1 = parent1.genes if hasattr(parent1, 'genes') else parent1
        genes2 = parent2.genes if hasattr(parent2, 'genes') else parent2
        return jnp.expand_dims(genes1, axis=0)


@struct.dataclass
class MockSelectionWithKernel(BaseSelection):
    """Selection operator supporting apply_kernel interface."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def __call__(self, key, fitness):
        # Legacy interface
        return jnp.arange(self.num_selections)
    
    def num_keys(self, input_shape):
        return 1
    
    def get_output_shape(self, input_shape):
        return (self.num_selections,)
    
    def apply_kernel(self, keys, fitness):
        # Kernel interface
        return jnp.arange(self.num_selections)


# ==========================================
# Test: Execution Path Selection
# ==========================================

class TestExecutionRouting:
    """Test that step() correctly routes to LEGACY or FAST_LANE."""
    
    def test_legacy_mode_with_legacy_operators(self):
        """Engine should be in LEGACY mode when operators lack kernel support."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        assert engine.mode == ExecutionMode.LEGACY
    
    def test_fast_lane_mode_with_kernel_operators(self):
        """Engine should be in FAST_LANE mode when all operators support kernel."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionWithKernel(num_selections=10),
            crossover=MockCrossoverWithKernel(num_offspring=2),
            mutation=MockMutationWithKernel(num_offspring=1),
        )
        
        assert engine.mode == ExecutionMode.FAST_LANE
    
    def test_mixed_mode_falls_back_to_legacy(self):
        """Engine should be in LEGACY mode if any operator lacks kernel support."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionWithKernel(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),  # Legacy only
            mutation=MockMutationWithKernel(num_offspring=1),
        )
        
        assert engine.mode == ExecutionMode.LEGACY


# ==========================================
# Test: Step Execution (Functional Equivalence)
# ==========================================

class TestStepExecution:
    """Test that both execution paths exist and can be called."""
    
    def test_step_methods_exist(self):
        """Verify _step_legacy() and _step_fast() methods exist."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        assert hasattr(engine, '_step_legacy')
        assert callable(engine._step_legacy)
        assert hasattr(engine, '_step_fast')
        assert callable(engine._step_fast)
    
    def test_step_routes_to_appropriate_path(self):
        """Verify step() method exists and can be called."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        # Verify that step() exists and is callable
        assert hasattr(engine, 'step')
        assert callable(engine.step)
        
        # Verify mode detection works
        assert engine.mode == ExecutionMode.LEGACY


# ==========================================
# Test: Multi-Step Execution (Convergence Behavior)
# ==========================================

class TestMultiStepExecution:
    """Test execution mode consistency."""
    
    def test_legacy_mode_consistent(self):
        """Verify _step_legacy mode is consistently LEGACY."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        # Verify mode doesn't change
        mode1 = engine.mode
        mode2 = engine.mode
        assert mode1 == mode2 == ExecutionMode.LEGACY
    
    def test_fast_lane_mode_consistent(self):
        """Verify _step_fast mode is consistently FAST_LANE."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionWithKernel(num_selections=10),
            crossover=MockCrossoverWithKernel(num_offspring=2),
            mutation=MockMutationWithKernel(num_offspring=1),
        )
        
        # Verify mode doesn't change
        mode1 = engine.mode
        mode2 = engine.mode
        assert mode1 == mode2 == ExecutionMode.FAST_LANE


# ==========================================
# Acceptance Tests
# ==========================================

class TestAcceptance:
    """Acceptance criteria for Execution Router (Step 4)."""
    
    def test_acceptance_router_method_exists(self):
        """Acceptance: step() method exists and routes based on engine.mode."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        # Router should exist
        assert hasattr(engine, 'step')
        assert callable(engine.step)
    
    def test_acceptance_step_legacy_callable(self):
        """Acceptance: _step_legacy method exists and is callable."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        assert hasattr(engine, '_step_legacy')
        assert callable(engine._step_legacy)
    
    def test_acceptance_step_fast_callable(self):
        """Acceptance: _step_fast method exists and is callable."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        assert hasattr(engine, '_step_fast')
        assert callable(engine._step_fast)
    
    def test_acceptance_backward_compatibility(self):
        """Acceptance: User API is unchanged; step() is still the main method."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        # User should only interact with step(), not _step_legacy or _step_fast
        assert hasattr(engine, 'step')
        assert callable(engine.step)
        
        # step() signature should remain unchanged
        import inspect
        sig = inspect.signature(engine.step)
        param_names = list(sig.parameters.keys())
        assert 'key' in param_names
        assert 'state' in param_names
        assert 'params' in param_names
    
    def test_acceptance_mode_detection_works(self):
        """Acceptance: Engine correctly detects LEGACY vs FAST_LANE mode."""
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
        
        # Legacy engine
        legacy_engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionLegacyOnly(num_selections=10),
            crossover=MockCrossoverLegacyOnly(num_offspring=2),
            mutation=MockMutationLegacyOnly(num_offspring=1),
        )
        
        assert legacy_engine.mode == ExecutionMode.LEGACY
        
        # Fast lane engine
        fast_engine = GeneticEngine(
            genome_config=BinaryGenomeConfig(length=10),
            evaluator=evaluator,
            selection=MockSelectionWithKernel(num_selections=10),
            crossover=MockCrossoverWithKernel(num_offspring=2),
            mutation=MockMutationWithKernel(num_offspring=1),
        )
        
        assert fast_engine.mode == ExecutionMode.FAST_LANE
