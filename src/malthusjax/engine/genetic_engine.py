"""
Standard Genetic Algorithm Engine.
Implements a modular, extensible evolutionary loop with callback support.
"""
import functools
import time
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from typing import Any, Tuple, Optional

from .base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams, AbstractGenerationOutput, AbstractHook
from ..operators.base import BaseMutation, BaseCrossover, BaseSelection
from ..core.fitness.base import BaseEvaluator
from ..core.base import BasePopulation
from ..core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..core.genome.categorical_genome import CategoricalGenomeConfig, CategoricalPopulation

# --- Host-Side Callback for Progress Bar ---
def _host_progress_callback(gen, best_fit):
    """
    This function runs on the CPU (Python) even when called from JIT code.
    Perfect for tqdm or printing.
    """
    print(f"Gen {gen}: Best Fitness = {best_fit:.4f}")

@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """Configuration for Genetic Engine."""
    pass

@struct.dataclass
class GeneticGenerationOutput(AbstractGenerationOutput):
    """KPIs returned by Genetic Engine."""
    pass

@struct.dataclass
class GeneticEngine(AbstractEngine):
    """
    A modular, extensible Genetic Engine.
    
    The 'step' function acts as a conductor, calling these replaceable parts:
    1. _select_elites()
    2. _select_parents()
    3. _create_offspring()
    4. _merge_and_evaluate()
    5. _update_hall_of_fame()
    6. _execute_hooks()
    """
    # Core Components
    genome_config: Any  # Configuration for genome creation
    evaluator: BaseEvaluator
    selection: BaseSelection
    crossover: BaseCrossover
    mutation: BaseMutation
    
    # Hooks (State Modifiers like Adaptive Mutation)
    hooks: Tuple[AbstractHook] = struct.field(default_factory=tuple)
    
    # Compilation cache (not part of pytree)
    _compiled_evolution_fn: Optional[Any] = struct.field(pytree_node=False, default=None)
    _compiled_for_params: Optional[Any] = struct.field(pytree_node=False, default=None)
    
    # Debug Config
    enable_progress_bar: bool = struct.field(pytree_node=False, default=False)
    
    # ==========================================
    # 1. TEMPLATE METHODS (Override these!)
    # ==========================================

    def _select_elites(self, population: BasePopulation, n_elites: int) -> Any:
        """Preserve the top K individual GENES."""
        # Note: We return genes, not population, for easier merging
        _, elite_indices = jax.lax.top_k(population.fitness, n_elites)
        # Slicing returns a Population, we extract .genes
        return population[elite_indices].genes

    def _select_parents(self, key: chex.Array, population: BasePopulation) -> BasePopulation:
        """Select parents for reproduction."""
        # Simple implementation: Select N parents where N = Pop Size
        # FIX: Use fitness_values from state if available, otherwise fallback to population.fitness
        # But here we receive population object which might not have updated fitness if we separate it
        # In the new paradigm, fitness is stored in state.fitness_values
        # However, _select_parents signature takes population.
        # We should update the signature or assume population has fitness property
        
        # For now, we assume population object has the fitness property populated
        # This is ensured by _merge_and_evaluate which returns a population with fitness
        indices = self.selection(key, population.fitness)
        return population[indices]

    def _create_offspring(self, key: chex.Array, parents: BasePopulation) -> Any:
        """Apply Variation (Crossover -> Mutation)."""
        k_cross, k_mut = jar.split(key)
        
        # A. Pairing - create pairs of parents for crossover
        p1 = parents
        # Permute parents to create random pairs
        k_perm, k_cross = jar.split(k_cross)
        p2_indices = jar.permutation(k_perm, jnp.arange(len(parents)))
        p2 = parents[p2_indices]
        
        # Determine config to use (prefer population config if available)
        op_config = getattr(parents, 'config', self.genome_config)
        
        # B. Apply crossover to each parent pair
        # crossover expects individual genomes, not batched populations
        # Use vmap to apply crossover to each pair
        def crossover_pair(key_i, genome1, genome2):
            return self.crossover(key_i, genome1, genome2, op_config)
        
        # Generate keys for each parent pair
        cross_keys = jar.split(k_cross, len(parents))
        
        # Apply crossover using vmap over parent pairs
        # Result shape: (num_parents, num_offspring, ...genome_shape)
        offspring_batched = jax.vmap(crossover_pair)(cross_keys, p1.genes, p2.genes)
        
        # Reshape to flatten parent and offspring dimensions
        # From (num_parents, num_offspring, ...genome_shape) 
        # To (num_parents * num_offspring, ...genome_shape)
        def flatten_first_two_dims(x):
            # x has shape (num_parents, num_offspring, ...genome_shape)
            return x.reshape(-1, *x.shape[2:])
        
        offspring_genes = jax.tree_util.tree_map(flatten_first_two_dims, offspring_batched)
        
        # C. Apply mutation to each offspring
        def mutate_single(key_i, genome):
            return self.mutation(key_i, genome, op_config)
        
        # Generate keys for each offspring
        mut_keys = jar.split(k_mut, jax.tree_util.tree_leaves(offspring_genes)[0].shape[0])
        
        # Apply mutation using vmap
        # Result shape: (num_offspring_total, num_mutants, ...genome_shape)
        mutants_batched = jax.vmap(mutate_single)(mut_keys, offspring_genes)
        
        # Flatten mutation batch dimension
        mutant_genes = jax.tree_util.tree_map(flatten_first_two_dims, mutants_batched)
        
        return mutant_genes

    def _merge_and_evaluate(self, elites_genes: Any, mutant_genes: Any, original_pop: BasePopulation) -> Tuple[BasePopulation, chex.Array]:
        """Combine elites and mutants, truncate to size, and evaluate."""
        pop_size = len(original_pop)
        
        # Infer num_elites from the array shape
        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        
        num_mutants_needed = pop_size - num_elites
        
        # Truncate mutants (take the first N needed)
        # We rely on tree_map to slice all arrays in the genome
        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants_needed], mutant_genes)
        
        # Concatenate Elites + Mutants
        next_genes = jax.tree_util.tree_map(
            lambda e, m: jnp.concatenate([e, m], axis=0),
            elites_genes,
            mutants_keep
        )
        
        # Wrap & Eval
        # We reuse the original population object to wrap the new genes (preserves type)
        unevaluated_pop = original_pop.replace(
            genes=next_genes,
            fitness=jnp.full((pop_size,), -jnp.inf)
        )
        
        # Delegate to Evaluator for vectorization (NEW paradigm)
        evaluated_pop = self.evaluator.evaluate_population(unevaluated_pop)
        fitness_values = evaluated_pop.fitness
        
        # Return both population (with updated fitness) and the fitness array
        return evaluated_pop, fitness_values

    def _update_hall_of_fame(self, state: AbstractEvolutionState, new_pop: BasePopulation) -> Tuple[Any, float, int]:
        """Update global best and stagnation counter."""
        best_idx = jnp.argmax(new_pop.fitness)
        curr_best_fit = new_pop.fitness[best_idx]
        curr_best_genome = new_pop[best_idx] # Returns Genome object via __getitem__

        is_new_record = curr_best_fit > state.best_fitness
        
        # Conditionally update best genome
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new_record, new, old),
            state.best_genome,
            curr_best_genome
        )
        new_best_fit = jnp.maximum(state.best_fitness, curr_best_fit)
        
        # Reset stagnation if new record, else increment
        new_stagnation = jnp.where(is_new_record, 0, state.stagnation_counter + 1)
        
        return new_best_genome, new_best_fit, new_stagnation

    def _execute_hooks(self, state: AbstractEvolutionState, params: AbstractEngineParams) -> AbstractEvolutionState:
        """Run all registered hooks (e.g., adaptive mutation)."""
        new_state = state
        for hook in self.hooks:
            new_state = hook(new_state, params)
        return new_state

    # ==========================================
    # 3. INITIALIZATION & STEP
    # ==========================================
    
    def init_state(self, rng_key: chex.Array, params: GeneticEngineParams) -> AbstractEvolutionState:
        """Initialize state from configuration."""
        # 1. Create initial population
        init_pop_key, rng_key = jar.split(rng_key)
        
        # Determine population class from genome config type
        if isinstance(self.genome_config, BinaryGenomeConfig):
            population_class = BinaryPopulation
        elif isinstance(self.genome_config, RealGenomeConfig):
            population_class = RealPopulation
        elif isinstance(self.genome_config, CategoricalGenomeConfig):
            population_class = CategoricalPopulation
        else:
            raise ValueError(f"Unsupported genome config type: {type(self.genome_config)}")
        
        population = population_class.init_random(init_pop_key, self.genome_config, params.pop_size)

        # 2. Evaluate initial population using NEW paradigm
        evaluated_pop = self.evaluator.evaluate_population(population)
        fitness_values = evaluated_pop.fitness
        
        # 3. Find best
        opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted_fitness = fitness_values * opt_sign
        best_idx = jnp.argmax(adjusted_fitness)
        best_genome = evaluated_pop[best_idx].genes  # Index population, not genes
        
        # 4. Create state
        return AbstractEvolutionState(
            population=evaluated_pop,
            fitness_values=fitness_values,
            best_genome=best_genome,
            best_fitness=fitness_values[best_idx],
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0
        )

    def step(
        self, 
        key: chex.Array, 
        state: AbstractEvolutionState, 
        params: GeneticEngineParams
    ) -> Tuple[chex.Array, AbstractEvolutionState, GeneticGenerationOutput]:
        """
        The Master Loop.
        """
        k_sel, k_gen, k_next = jar.split(key, 3)
        
        # 1. Elitism
        elites_genes = self._select_elites(state.population, params.elitism)
        
        # 2. Selection
        parents = self._select_parents(k_sel, state.population)
        
        # 3. Variation
        mutants_genes = self._create_offspring(k_gen, parents)
        
        # 4. Merge & Eval
        new_pop, fitness_values = self._merge_and_evaluate(elites_genes, mutants_genes, state.population)
        
        # 5. Stats
        # Use JAX operations for optimization direction
        opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        
        # Find best in current generation
        adjusted_fitness = fitness_values * opt_sign
        current_best_idx = jnp.argmax(adjusted_fitness)
        current_best_fitness = fitness_values[current_best_idx]
        current_best_genome = new_pop[current_best_idx].genes  # Extract genes from population
        
        # Check improvement
        is_improvement = (current_best_fitness * opt_sign) > (state.best_fitness * opt_sign)
        
        # Update global bests
        new_best_fitness = jnp.where(is_improvement, current_best_fitness, state.best_fitness)
        new_best_genome = jax.tree_util.tree_map(
            lambda new, old: jnp.where(is_improvement, new, old),
            current_best_genome,
            state.best_genome
        )
        
        # Update stagnation
        new_stagnation = jnp.where(is_improvement, 0, state.stagnation_counter + 1)
        
        # 6. Create Intermediate State
        temp_state = state.replace(
            population=new_pop,
            fitness_values=fitness_values,
            best_genome=new_best_genome,
            best_fitness=new_best_fitness,
            stagnation_counter=new_stagnation,
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        # 7. Run Hooks
        final_state = self._execute_hooks(temp_state, params)
        
        # 8. Runtime Logging (Callback)
        if self.enable_progress_bar:
            jax.debug.callback(
                _host_progress_callback, 
                final_state.generation, 
                final_state.best_fitness
            )

        # 9. Metrics
        metrics = GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation
        )
        
        return k_next, final_state, metrics