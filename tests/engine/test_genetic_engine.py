import pytest
import chex
import jax
import jax.numpy as jnp

from malthusjax.engine.genetic_fastengine import GeneticEvolutionState

def test_01_init_state_baking(make_engine, prng_key):
    """Test if init_state correctly compiles the plan and bakes operators."""
    engine = make_engine(pop_size=100, elitism=2)
    state = engine.init_state(prng_key)

    # Check State Integrity
    assert isinstance(state, GeneticEvolutionState)
    assert state.generation == 0
    chex.assert_shape(state.population.fitness, (100,))

    # Verify Supply/Demand Logic (Pop 100 -> Pairs 50 -> Parents 100)
    rmap = state.resource_map
    assert rmap.selection.output_count == 98  # pop_size - elitism

    # Verify Baked Operators
    ops = state.operators
    assert ops.selection.num_selections == 98
    assert ops.crossover.input_length == 49
    assert ops.mutation.input_length == 98

@pytest.mark.slow
def test_02_step_execution(make_engine, prng_key):
    """Test a single manual step execution."""
    engine = make_engine(pop_size=100, genome_shape=(10,))
    state = engine.init_state(prng_key)

    jit_step = jax.jit(engine.step)
    final_state, metrics = jit_step(state)
    _ = final_state.best_fitness.block_until_ready()

    assert final_state.generation == 1
    chex.assert_shape(final_state.population.genes.values, (100, 10))
    chex.assert_tree_all_finite(final_state.best_fitness)

def test_02b_debug_step_execution(make_engine, prng_key):
    """Test the debug step helper on the fast engine."""
    engine = make_engine(pop_size=100, genome_shape=(10,))
    state = engine.init_state(prng_key)

    final_state, metrics = engine.debug_step(state)

    assert final_state.generation == 1
    chex.assert_shape(final_state.population.genes.values, (100, 10))
    assert metrics.generation == 1

def test_03_closed_loop_fusion(make_engine, prng_key):
    """Test Level 3 'Closed Loop' Compilation and Fusion."""
    engine = make_engine()
    state = engine.init_state(prng_key)

    hlo_text = engine.get_hlo_text(state, optimize=True, print_analysis=False)
    
    # Check for Fusion and Loops
    assert hlo_text.count("fusion") > 0, "Warning: No fusion detected."
    assert "while" in hlo_text, "Critical: Python loop was NOT compiled into XLA while loop."

@pytest.mark.parametrize("pop_size", [2, 7, 17, 200])
def test_odd_population_sizes(make_engine, prng_key, pop_size):
    """Test the ResourceMapper fix for odd population sizes."""
    engine = make_engine(pop_size=pop_size, elitism=1, num_generations=2)
    state = engine.init_state(prng_key)
    
    final_state, _, _ = engine.run(state, compile=False)
    chex.assert_shape(final_state.population.genes.values, (pop_size, 10))

def test_engine_execution_quality(make_engine, prng_key):
    """Test engine execution and output quality (no NaN/Inf, bounds respected)."""
    bounds = (-5.0, 5.0)
    engine = make_engine(pop_size=30, genome_shape=(5,), bounds=bounds, num_generations=2)
    state = engine.init_state(prng_key)
    final_state, _, _ = engine.run(state, compile=False)
    
    genes = final_state.population.genes.values
    fitness = final_state.population.fitness
    
    chex.assert_tree_all_finite(genes)
    chex.assert_tree_all_finite(fitness)
    chex.assert_tree_all_finite(final_state.best_fitness)
    
    within_bounds = jnp.all(genes >= bounds[0]) and jnp.all(genes <= bounds[1])
    assert within_bounds, "Genes violated bounds"
    
    best = jnp.min(fitness)
    mean = jnp.mean(fitness)
    assert float(best) <= float(mean) + 1e-5

def test_deterministic_with_same_seed(make_engine):
    """Test that engine is deterministic with the same seed."""
    def run_with_seed(seed):
        engine = make_engine(pop_size=20, num_generations=2)
        state = engine.init_state(jax.random.PRNGKey(seed))
        final_state, _, _ = engine.run(state, compile=False)
        return float(final_state.best_fitness)

    assert run_with_seed(42) == run_with_seed(42)
    assert run_with_seed(123) != run_with_seed(42)

@pytest.mark.slow
@pytest.mark.integration
def test_bbob_minimization_improves(make_engine, prng_key):
    """Test that minimization improves over generations."""
    engine = make_engine(pop_size=50, num_generations=10, maximize=False)
    state = engine.init_state(prng_key)
    
    best_history = [float(state.best_fitness)]
    for _ in range(10):
        state, output = engine.step(state)
        best_history.append(float(output.best_fitness))
        
    for i in range(1, len(best_history)):
        assert best_history[i] <= best_history[i - 1] + 1e-5

@pytest.mark.slow
@pytest.mark.integration
def test_maximization_monotonic_improvement_real(make_engine, prng_key):
    """Test that best fitness increases monotonically when maximizing (maximize=True)."""
    engine = make_engine(pop_size=50, num_generations=10, maximize=True)
    state = engine.init_state(prng_key)
    
    best_history = [float(state.best_fitness)]
    for _ in range(10):
        state, output = engine.step(state)
        best_history.append(float(output.best_fitness))
        
    for i in range(1, len(best_history)):
        assert best_history[i] <= best_history[i - 1] + 1e-5

@pytest.mark.slow
@pytest.mark.integration
def test_maximization_monotonic_improvement_binary(make_engine, prng_key):
    """Test that best fitness increases monotonically when maximizing with binary genomes."""
    engine = make_engine(pop_size=50, genome_type="binary", genome_shape=(20,), maximize=True)
    state = engine.init_state(prng_key)
    
    best_history = [float(state.best_fitness)]
    for _ in range(15):
        state, output = engine.step(state)
        best_history.append(float(output.best_fitness))
        
    for i in range(1, len(best_history)):
        assert best_history[i] <= best_history[i - 1] + 1e-5

@pytest.mark.parametrize("num_gens", [1, 50])
def test_edge_case_generations(make_engine, prng_key, num_gens):
    """Test extreme generation counts."""
    engine = make_engine(num_generations=num_gens)
    state = engine.init_state(prng_key)
    final_state, _, _ = engine.run(state, compile=False)
    assert final_state.generation == num_gens

@pytest.mark.parametrize("dim", [1, 20])
def test_edge_case_genome_dimensions(make_engine, prng_key, dim):
    """Test extreme genome dimensions."""
    engine = make_engine(pop_size=15, genome_shape=(dim,), num_generations=2)
    state = engine.init_state(prng_key)
    final_state, _, _ = engine.run(state, compile=False)
    chex.assert_shape(final_state.population.genes.values, (15, dim))

@pytest.mark.parametrize("mutation_rate", [0.0, 0.9])
def test_edge_case_mutation_rates(make_engine, prng_key, mutation_rate):
    """Test extreme mutation rates."""
    engine = make_engine(mutation_rate=mutation_rate, num_generations=2)
    state = engine.init_state(prng_key)
    final_state, _, _ = engine.run(state, compile=False)
    assert final_state.generation == 2
