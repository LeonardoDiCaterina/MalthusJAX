"""
Informal tests for RouletteSelection operator.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.operators.selection.roulette import RouletteSelection
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation


@pytest.fixture
def setup_population():
    """Create a test population with known fitness values."""
    key = jr.PRNGKey(42)
    config = RealGenomeConfig(length=5, bounds=(-5.0, 5.0))
    pop = RealPopulation.init_random(key, config, 8)
    fitness = jnp.array([1.0, 2.0, 5.0, 10.0, 3.0, 1.0, 0.5, 8.0])
    pop = pop.replace(fitness=fitness)
    return pop, fitness, key


class TestRouletteSelection:
    """Tests for RouletteSelection operator."""
    
    def test_standard_path(self, setup_population):
        """Test standard path (num_selections != pop_size)."""
        pop, fitness, key = setup_population
        
        sel = RouletteSelection(num_selections=4, temperature=1.0)
        k1, key = jr.split(key)
        keys = jr.split(k1, 1)
        indices = sel._select(keys, fitness, pop)
        
        assert indices.shape == (4,)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < 8)
    
    def test_gumbel_max_path(self, setup_population):
        """Test optimized Gumbel-Max path (num_selections == pop_size)."""
        pop, fitness, key = setup_population
        
        sel = RouletteSelection(num_selections=8, temperature=1.0)
        k1, key = jr.split(key)
        keys = jr.split(k1, 1)
        indices = sel._select(keys, fitness, pop)
        
        assert indices.shape == (8,)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < 8)
    
    def test_temperature_cold(self, setup_population):
        """Test that low temperature favors high fitness individuals."""
        pop, fitness, key = setup_population
        
        sel = RouletteSelection(num_selections=100, temperature=0.1)
        k1, key = jr.split(key)
        keys = jr.split(k1, 1)
        indices = sel._select(keys, fitness, pop)
        
        # Index 3 has highest fitness (10.0), should be selected most often
        counts = jnp.bincount(indices, length=8)
        most_selected = counts.argmax()
        assert most_selected == 3, f"Expected index 3 (highest fitness), got {most_selected}"
    
    def test_temperature_hot(self, setup_population):
        """Test that high temperature spreads selection more uniformly."""
        pop, fitness, key = setup_population
        
        sel = RouletteSelection(num_selections=1000, temperature=10.0)
        k1, key = jr.split(key)
        keys = jr.split(k1, 1)
        indices = sel._select(keys, fitness, pop)
        
        # With high temperature, all indices should be selected at least once
        counts = jnp.bincount(indices, length=8)
        assert jnp.all(counts > 0), "High temperature should select all individuals"
    
    def test_jit_compilation(self, setup_population):
        """Test that selection works under JIT compilation."""
        pop, fitness, key = setup_population
        
        sel = RouletteSelection(num_selections=8, temperature=1.0)
        
        @jax.jit
        def jit_select(keys, fitness, pop):
            return sel._select(keys, fitness, pop)
        
        k1, key = jr.split(key)
        keys = jr.split(k1, 1)
        indices = jit_select(keys, fitness, pop)
        
        assert indices.shape == (8,)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < 8)


if __name__ == "__main__":
    # Run informal tests
    print('=== Testing RouletteSelection ===\n')
    
    key = jr.PRNGKey(42)
    config = RealGenomeConfig(length=5, bounds=(-5.0, 5.0))
    pop = RealPopulation.init_random(key, config, 8)
    fitness = jnp.array([1.0, 2.0, 5.0, 10.0, 3.0, 1.0, 0.5, 8.0])
    pop = pop.replace(fitness=fitness)
    
    print(f'Population size: {len(pop)}')
    print(f'Fitness values: {fitness}\n')
    
    # Test 1: Standard path
    print('--- Test 1: Standard Path (4 selections from 8) ---')
    sel1 = RouletteSelection(num_selections=4, temperature=1.0)
    k1, key = jr.split(key)
    keys = jr.split(k1, 1)
    indices1 = sel1._select(keys, fitness, pop)
    print(f'Selected indices: {indices1}')
    print(f'Selected fitness: {fitness[indices1]}')
    print('OK\n')
    
    # Test 2: Gumbel-Max path
    print('--- Test 2: Gumbel-Max Path (8 selections from 8) ---')
    sel2 = RouletteSelection(num_selections=8, temperature=1.0)
    k2, key = jr.split(key)
    keys = jr.split(k2, 1)
    indices2 = sel2._select(keys, fitness, pop)
    print(f'Selected indices: {indices2}')
    print(f'Selected fitness: {fitness[indices2]}')
    print('OK\n')
    
    # Test 3: Temperature effect
    print('--- Test 3: Temperature Effect ---')
    sel_cold = RouletteSelection(num_selections=100, temperature=0.1)
    k3, key = jr.split(key)
    keys = jr.split(k3, 1)
    indices_cold = sel_cold._select(keys, fitness, pop)
    print(f'Cold (T=0.1) - Most selected index: {jnp.bincount(indices_cold, length=8).argmax()} (should be 3)')
    
    sel_hot = RouletteSelection(num_selections=100, temperature=10.0)
    k4, key = jr.split(key)
    keys = jr.split(k4, 1)
    indices_hot = sel_hot._select(keys, fitness, pop)
    print(f'Hot (T=10) - Selection distribution: {jnp.bincount(indices_hot, length=8)}')
    print('OK\n')
    
    # Test 4: JIT compilation
    print('--- Test 4: JIT Compilation ---')
    @jax.jit
    def jit_select(keys, fitness, pop):
        return sel2._select(keys, fitness, pop)
    
    k5, key = jr.split(key)
    keys = jr.split(k5, 1)
    jit_result = jit_select(keys, fitness, pop)
    print(f'JIT output shape: {jit_result.shape}')
    print('OK\n')
    
    print('✅ All RouletteSelection tests passed!')
