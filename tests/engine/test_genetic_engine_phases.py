"""
Comprehensive phase-level tests for GeneticEngine.
"""
import pytest
import chex
import jax
import jax.numpy as jnp
from malthusjax.engine.genetic_fastengine import GeneticEngineParams

def test_entropy_allocation_returns_four_keys(make_engine, prng_key):
    engine = make_engine(pop_size=50)
    state = engine.init_state(prng_key)
    
    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)
    assert isinstance(k_sel, jax.Array)
    assert isinstance(k_cross, jax.Array)
    assert isinstance(k_mut, jax.Array)
    assert isinstance(k_next, jax.Array)
    
    assert len(k_sel.shape) == 2
    assert len(k_cross.shape) == 2
    assert len(k_mut.shape) == 2
    assert k_next.shape == (2,)

def test_selection_returns_elite_and_parents(make_engine, prng_key):
    engine = make_engine(pop_size=40, elitism=3)
    state = engine.init_state(prng_key)
    key_sel = jax.random.fold_in(state.rng_key, 0)
    
    elites, parent_indices = engine._selection_phase(
        key_sel, state.population, state.operators, engine.engine_params
    )
    
    chex.assert_shape(elites.values, (3, 10))
    expected_parents = 2 * ((40 - 3 + 1) // 2)
    assert parent_indices.shape[0] == expected_parents
    assert jnp.all(parent_indices >= 0)
    assert jnp.all(parent_indices < 40)

def test_elite_genes_are_best_fitness(make_engine, prng_key):
    engine = make_engine(pop_size=40, elitism=3)
    state = engine.init_state(prng_key)
    key_sel = jax.random.fold_in(state.rng_key, 0)
    
    elites, _ = engine._selection_phase(
        key_sel, state.population, state.operators, engine.engine_params
    )
    top_k_indices = jnp.argsort(state.population.fitness)[:3]
    top_k_genes = state.population[top_k_indices].genes
    
    chex.assert_trees_all_close(elites.values, top_k_genes.values)

def test_reproduction_produces_correct_population_size(make_engine, prng_key):
    engine = make_engine(pop_size=30, elitism=2)
    state = engine.init_state(prng_key)
    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)
    
    _, parent_indices = engine._selection_phase(
        k_sel, state.population, state.operators, engine.engine_params
    )
    
    final_pop = engine._reproduction_phase(
        k_cross, k_mut, parent_indices, state.population, state.operators, state.resource_map
    )
    
    chex.assert_shape(final_pop.genes.values, (state.resource_map.mutation.output_count, 10))

def test_reproduction_produces_different_offspring(make_engine, prng_key):
    engine = make_engine(pop_size=30, elitism=2)
    state = engine.init_state(prng_key)
    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)
    
    _, parent_indices = engine._selection_phase(
        k_sel, state.population, state.operators, engine.engine_params
    )
    
    offspring_pop = engine._reproduction_phase(
        k_cross, k_mut, parent_indices, state.population, state.operators, state.resource_map
    )
    
    offspring_genes = offspring_pop.genes.values
    parent_genes = state.population[parent_indices].genes.values
    
    identical_count = jnp.sum(jnp.all(offspring_genes == parent_genes, axis=-1))
    assert identical_count < len(offspring_genes) * 0.5

def test_merge_preserves_elite_at_top(make_engine, prng_key):
    engine = make_engine(pop_size=40, elitism=4)
    state = engine.init_state(prng_key)
    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)
    
    elites, parent_indices = engine._selection_phase(
        k_sel, state.population, state.operators, engine.engine_params
    )
    
    mutants = engine._reproduction_phase(
        k_cross, k_mut, parent_indices, state.population, state.operators, state.resource_map
    )
    
    merged_genes = engine._merge(elites, mutants.genes, state)
    chex.assert_shape(merged_genes.values, (40, 10))
    chex.assert_trees_all_close(merged_genes.values[:4], elites.values)

def test_evaluation_produces_fitness_for_all(make_engine, prng_key):
    engine = make_engine(pop_size=30, elitism=2)
    state = engine.init_state(prng_key)
    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)
    
    elites, parent_indices = engine._selection_phase(
        k_sel, state.population, state.operators, engine.engine_params
    )
    
    mutants = engine._reproduction_phase(
        k_cross, k_mut, parent_indices, state.population, state.operators, state.resource_map
    )
    new_genes = engine._merge(elites, mutants.genes, state)
    evaluated_pop = engine._evaluate(new_genes, state)
    
    chex.assert_shape(evaluated_pop.fitness, (30,))
    chex.assert_tree_all_finite(evaluated_pop.fitness)

def test_step_updates_best_fitness(make_engine, prng_key):
    engine = make_engine(pop_size=30, elitism=2)
    state = engine.init_state(prng_key)
    updated_state, _ = engine.step(state)
    
    assert updated_state.generation == state.generation + 1
    assert updated_state.best_fitness is not None
    assert updated_state.best_genome is not None

def test_step_preserves_best_when_no_improvement(make_engine, prng_key):
    engine = make_engine(pop_size=30, elitism=2)
    state = engine.init_state(prng_key)
    state_after_step, _ = engine.step(state)
    state_after_two, _ = engine.step(state_after_step)
    
    assert float(state_after_two.best_fitness) <= float(state_after_step.best_fitness) + 1e-5
