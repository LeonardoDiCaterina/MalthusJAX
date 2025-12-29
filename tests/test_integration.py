"""
Integration tests for the full evolutionary pipeline.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

# 1. Import Core Components (Exposed at Top Level)
from malthusjax import (
    BinaryGenome, BinaryGenomeConfig, BinaryPopulation,
    BinarySumEvaluator, BinarySumConfig
)

from malthusjax.operators.selection import TournamentSelection
from malthusjax.operators.mutation import BitFlipMutation, ScrambleMutation

# 2. Import The Library Namespace (For Operators)
import malthusjax as mjx

class TestBinaryEvolutionPipeline:
    def test_single_generation_binary(self, rng_key):
        # Setup
        binary_genome_config = BinaryGenomeConfig(length=10)
        pop_size = 10
        k1, k2, k3, k4 = jr.split(rng_key, 4)
        
        # 1. Init
        population = BinaryPopulation.init_random(k1, binary_genome_config, pop_size)
        assert len(population) == pop_size

        # 2. Eval
        eval_config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=eval_config, data=None)
        evaluated_pop = evaluator.evaluate_population(population)
        fitness = evaluated_pop.fitness

        # 3. Select (TournamentSelection overrides __call__ to take raw fitness)
        selector = TournamentSelection(num_selections=pop_size, tournament_size=3)
        selected_indices = selector(k2, fitness)
        parents = population[selected_indices]

        # 4. Crossover (Batch-First) - requires pre-split keys
        half = pop_size // 2
        p1 = parents[:half]  # Returns Population slice (not raw genes)
        p2 = parents[half:]
        
        crossover = mjx.crossover.UniformCrossover(num_offspring=2, crossover_rate=0.5)
        # Crossover needs keys: (num_pairs * num_offspring * keys_per_op, 2)
        num_cross_keys = half * crossover.num_offspring * crossover.num_keys_per_atomic_operation
        cross_keys = jr.split(k3, num_cross_keys)
        
        # Pass populations (not .genes) - crossover calls p1.spawn_offspring()
        offspring_pop = crossover(cross_keys, p1, p2, binary_genome_config)
        
        # offspring_pop is a Population; extract bits
        flat_bits = offspring_pop.genes.bits
        
        assert flat_bits.shape == (pop_size, binary_genome_config.length)

        # 5. Mutation (Batch-First) - requires pre-split keys and population object
        mutator = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
        # Mutation needs keys: (pop_size * num_offspring * keys_per_op, 2)
        num_mut_keys = pop_size * mutator.num_offspring * mutator.num_keys_per_atomic_operation
        mut_keys = jr.split(k4, num_mut_keys)
        
        # Pass population (not raw genes) - mutation calls population.spawn_offspring()
        mutated_pop = mutator(mut_keys, offspring_pop, binary_genome_config)
        
        # mutated_pop is a Population; extract bits
        final_bits = mutated_pop.genes.bits
        
        assert final_bits.shape == (pop_size, binary_genome_config.length)

    @pytest.mark.jit
    def test_jit_compiled_binary_generation(self, rng_key):
        """Verify JIT compilation of the loop."""
        config = BinaryGenomeConfig(length=10)
        pop_size = 20
        half = pop_size // 2
        
        # Bake operators
        selector = TournamentSelection(num_selections=pop_size, tournament_size=3)
        crossover = mjx.crossover.UniformCrossover(num_offspring=2, crossover_rate=0.5)
        mutator = BitFlipMutation(num_offspring=1, mutation_rate=0.01)
        
        # Pre-compute key counts for static allocation
        num_cross_keys = half * crossover.num_offspring * crossover.num_keys_per_atomic_operation
        num_mut_keys = pop_size * mutator.num_offspring * mutator.num_keys_per_atomic_operation

        @jax.jit
        def evolution_step(key, current_bits, fitness):
            k_sel, k_cross, k_mut = jr.split(key, 3)
            
            # Select (TournamentSelection takes raw fitness)
            indices = selector(k_sel, fitness)
            selected_bits = current_bits[indices]
            
            # Crossover needs Population objects (not raw genomes)
            cross_keys = jr.split(k_cross, num_cross_keys)
            p1_genes = BinaryGenome(bits=selected_bits[:half])
            p2_genes = BinaryGenome(bits=selected_bits[half:])
            # Create Population wrappers - crossover calls spawn_offspring()
            p1_pop = BinaryPopulation(
                genes=p1_genes, 
                fitness=jnp.zeros(half), 
                config=config
            )
            p2_pop = BinaryPopulation(
                genes=p2_genes, 
                fitness=jnp.zeros(half), 
                config=config
            )
            
            off_pop = crossover(cross_keys, p1_pop, p2_pop, config)
            off_bits = off_pop.genes.bits
            
            # Mutate needs Population object
            mut_keys = jr.split(k_mut, num_mut_keys)
            off_genes = BinaryGenome(bits=off_bits)
            off_pop_for_mut = BinaryPopulation(
                genes=off_genes,
                fitness=jnp.zeros(pop_size),
                config=config
            )
            mut_pop = mutator(mut_keys, off_pop_for_mut, config)
            
            return mut_pop.genes.bits

        # Run
        pop = BinaryPopulation.init_random(rng_key, config, pop_size)
        fitness = jnp.zeros(pop_size)
        new_bits = evolution_step(rng_key, pop.genes.bits, fitness)
        
        assert new_bits.shape == (pop_size, config.length)

    def test_large_scale_integration(self, rng_key):
        pop_size = 100
        length = 50
        config = BinaryGenomeConfig(length=length)
        pop = BinaryPopulation.init_random(rng_key, config, pop_size)
        
        assert pop.genes.bits.shape == (pop_size, length)
        
    def test_multi_operator_compatibility(self, rng_key):
        config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config)
        
        # Operators
        op1 = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
        op2 = ScrambleMutation(num_offspring=1, mutation_rate=1.0)
        
        # Use _mutate_one for single genome testing
        k1, k2 = jr.split(rng_key)
        
        # _mutate_one expects keys with shape (num_keys_per_atomic_operation, 2)
        keys_1 = jr.split(k1, op1.num_keys_per_atomic_operation)
        mutated1 = op1._mutate_one(keys_1, genome, config)
        
        # Chain - second mutation
        keys_2 = jr.split(k2, op2.num_keys_per_atomic_operation)
        mutated2 = op2._mutate_one(keys_2, mutated1, config)
        
        assert mutated2.bits.shape == (config.length,)