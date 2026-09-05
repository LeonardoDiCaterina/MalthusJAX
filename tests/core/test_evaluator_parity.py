"""Parity tests between legacy monolithic evaluators and new composable architecture."""

import chex
import jax
import jax.numpy as jnp

from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.fitness.composable.base import ScalarOutput
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator

# Legacy Evaluators & Configs
from malthusjax.core.fitness.real_evaluators import (
    SphereEvaluator, SphereConfig,
    GriewankEvaluator, GriewankConfig,
    BoxEvaluator, BoxConfig,
)
from malthusjax.core.fitness.tsp_evaluator import TSPEvaluator, TSPConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig

# Composable Environments
from malthusjax.core.fitness.composable.environments import (
    SphereEnv, GriewankEnv, BoxEnv, TSPEnv, BBOBEnv
)


from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

def _generate_population(pop_size: int = 100, genome_length: int = 10, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    config = RealGenomeConfig(shape=(genome_length,), bounds=(-5.0, 5.0))
    genes = RealGenome.create_population(key, config=config, pop_size=pop_size)
    return BasePopulation(genes=genes, fitness=jnp.zeros(pop_size))


def test_sphere_parity():
    """Test parity between legacy SphereEvaluator and Composable SphereEnv."""
    pop = _generate_population(genome_length=10)
    
    # 1. Minimize (maximize=False)
    # The legacy evaluator had a bug where it negated the score for minimize.
    # The new one handles it correctly in ScalarOutput (returns raw score for minimize).
    legacy_min = SphereEvaluator(config=SphereConfig(maximize=False))
    pop_legacy_min = legacy_min.evaluate_population(pop)
    
    composable_min = OptimizationEvaluator(
        env=SphereEnv(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )
    pop_composable_min = composable_min.evaluate_population(pop)
    
    # Asserting the fix: legacy actually inverted the value, composable returns the true positive sum
    chex.assert_trees_all_close(pop_legacy_min.fitness, -pop_composable_min.fitness)

    # 2. Maximize (maximize=True)
    # Both should return the exact same positive sum (legacy returned raw, composable negates it? Wait!)
    # Actually, legacy Sphere returned positive raw value when maximize=True.
    # Composable returns NEGATIVE raw value when maximize=True (so engine can minimize it).
    legacy_max = SphereEvaluator(config=SphereConfig(maximize=True))
    pop_legacy_max = legacy_max.evaluate_population(pop)
    
    composable_max = OptimizationEvaluator(
        env=SphereEnv(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=True)
    )
    pop_composable_max = composable_max.evaluate_population(pop)
    
    chex.assert_trees_all_close(pop_legacy_max.fitness, -pop_composable_max.fitness)


def test_griewank_parity():
    """Test parity between legacy GriewankEvaluator and Composable GriewankEnv."""
    pop = _generate_population(genome_length=10)
    
    legacy_min = GriewankEvaluator(config=GriewankConfig(maximize=False))
    pop_legacy_min = legacy_min.evaluate_population(pop)
    
    composable_min = OptimizationEvaluator(
        env=GriewankEnv(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )
    pop_composable_min = composable_min.evaluate_population(pop)
    
    # Asserting the fix: legacy inverted the value
    chex.assert_trees_all_close(pop_legacy_min.fitness, -pop_composable_min.fitness)


def test_box_parity():
    """Test parity between legacy BoxEvaluator and Composable BoxEnv."""
    pop = _generate_population(genome_length=10)
    
    target = jnp.zeros(10)
    lower = jnp.full(10, -2.5)
    upper = jnp.full(10, 2.5)
    
    legacy_min = BoxEvaluator(config=BoxConfig(target_point=target, box_bounds=(lower, upper), objective_type="distance", maximize=False))
    pop_legacy_min = legacy_min.evaluate_population(pop)
    
    composable_min = OptimizationEvaluator(
        env=BoxEnv(target_point=target, box_bounds=(lower, upper), objective_type="distance"),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )
    pop_composable_min = composable_min.evaluate_population(pop)
    
    # Asserting the fix: legacy inverted the value
    chex.assert_trees_all_close(pop_legacy_min.fitness, -pop_composable_min.fitness, atol=1e-5)


def test_tsp_parity():
    """Test parity between legacy TSPEvaluator and Composable TSPEnv.
    TSP legacy evaluator handled 'maximize' correctly.
    """
    pop = _generate_population(genome_length=10)
    
    key = jax.random.PRNGKey(42)
    coords = jax.random.uniform(key, (10, 2))
    diff = coords[:, jnp.newaxis, :] - coords[jnp.newaxis, :, :]
    distance_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    
    # 1. Minimize (maximize=False)
    legacy_min = TSPEvaluator(config=TSPConfig(maximize=False), data=distance_matrix)
    pop_legacy_min = legacy_min.evaluate_population(pop)
    
    composable_min = OptimizationEvaluator(
        env=TSPEnv(distance_matrix=distance_matrix),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )
    pop_composable_min = composable_min.evaluate_population(pop)
    
    chex.assert_trees_all_close(pop_legacy_min.fitness, pop_composable_min.fitness)

    # 2. Maximize (maximize=True)
    legacy_max = TSPEvaluator(config=TSPConfig(maximize=True), data=distance_matrix)
    pop_legacy_max = legacy_max.evaluate_population(pop)
    
    composable_max = OptimizationEvaluator(
        env=TSPEnv(distance_matrix=distance_matrix),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=True)
    )
    pop_composable_max = composable_max.evaluate_population(pop)
    
    chex.assert_trees_all_close(pop_legacy_max.fitness, pop_composable_max.fitness)


def test_bbob_parity():
    """Test parity between legacy BBOBEvaluator and Composable BBOBEnv."""
    pop = _generate_population(genome_length=10)
    
    legacy_min = BBOBEvaluator.create(config=BBOBConfig(fn_name="sphere", num_dims=10, seed=1))
    # Override maximize for testing (it defaults to False)
    legacy_min = legacy_min.replace(config=legacy_min.config.replace(maximize=False))
    pop_legacy_min = legacy_min.evaluate_population(pop)
    
    composable_min = OptimizationEvaluator(
        env=BBOBEnv.create(fn_name="sphere", num_dims=10, seed=1),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False)
    )
    pop_composable_min = composable_min.evaluate_population(pop)
    
    chex.assert_trees_all_close(pop_legacy_min.fitness, pop_composable_min.fitness, atol=1e-5)
    
    # Test Maximize
    legacy_max = legacy_min.replace(config=legacy_min.config.replace(maximize=True))
    pop_legacy_max = legacy_max.evaluate_population(pop)
    
    composable_max = OptimizationEvaluator(
        env=BBOBEnv.create(fn_name="sphere", num_dims=10, seed=1),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=True)
    )
    pop_composable_max = composable_max.evaluate_population(pop)
    
    chex.assert_trees_all_close(pop_legacy_max.fitness, pop_composable_max.fitness, atol=1e-5)
