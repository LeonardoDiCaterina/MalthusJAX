"""
Standard Genetic Algorithm Engine.
Implements a modular, extensible evolutionary loop with 'Full Access' component design.
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
    # Add any specific parameters here if needed, otherwise inherits basics
    pass

@struct.dataclass
class GeneticGenerationOutput(AbstractGenerationOutput):
    """KPIs returned by Genetic Engine."""
    pass


@struct.dataclass
class GeneticEngine(AbstractEngine):
    """
    A modular, extensible Genetic Engine.
    
    Architecture: 'Full Access' Components
    Every internal method receives (key, state, params) to allow for 
    adaptive logic (e.g. changing mutation rates based on state.stagnation)
    and easy unit testing.
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
    # 1. MODULAR COMPONENT METHODS
    # ==========================================

    def _select_elites(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> chex.ArrayTree:
        """
        Preserve the top individuals.
        """
        # FIX: Use 'params.elitism' (standard name), not 'n_elites'
        _, elite_indices = jax.lax.top_k(state.population.fitness, params.elitism)
        return state.population[elite_indices].genes

    def _select_parents(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> BasePopulation:
        """
        Select parents for reproduction.
        """
        indices = self.selection(key, state.population.fitness)
        
        # FIX: This is the specific line that prevents the dimension error!
        # Ensures indices are (N,) instead of (N, 1) or (N, 1, 1)
        indices = indices.flatten()
        
        return state.population[indices]

    def _create_offspring(self, key: chex.PRNGKey, parents: BasePopulation, state: AbstractEvolutionState, params: GeneticEngineParams) -> chex.ArrayTree:
        """
        Apply Variation (Crossover -> Mutation) with automatic flattening.
        Handles cases where operators return multiple offspring per parent.
        """
        k_cross, k_mut = jar.split(key)
        
        # A. Pairing
        p1 = parents
        k_perm, k_cross = jar.split(k_cross)
        p2_indices = jar.permutation(k_perm, jnp.arange(len(parents)))
        p2 = parents[p2_indices]
        
        op_config = getattr(parents, 'config', self.genome_config)
        
        # B. Crossover
        def crossover_pair(k, g1, g2):
            return self.crossover(k, g1, g2, op_config)
            
        cross_keys = jar.split(k_cross, len(parents))
        # Shape: (N_Pairs, Num_Offspring_Cross, Genome_Len) -> e.g., (100, 2, 100)
        offspring_genes = jax.vmap(crossover_pair)(cross_keys, p1.genes, p2.genes)
        
        # FLATTEN STEP 1: Collapse the first two dimensions
        # (100, 2, 100) -> (200, 100)
        def flatten_batch(x):
            return x.reshape(-1, *x.shape[2:])
            
        offspring_genes = jax.tree_util.tree_map(flatten_batch, offspring_genes)
        
        # C. Mutation
        def mutate_single(k, g):
            return self.mutation(k, g, op_config)
            
        # We now have more genes (e.g., 200), so we need more keys
        num_current_offspring = jax.tree_util.tree_leaves(offspring_genes)[0].shape[0]
        mut_keys = jar.split(k_mut, num_current_offspring)
        
        # Shape: (N_Offspring, Num_Offspring_Mut, Genome_Len) -> e.g., (200, 2, 100)
        mutant_genes = jax.vmap(mutate_single)(mut_keys, offspring_genes)
        
        # FLATTEN STEP 2: Collapse again
        # (200, 2, 100) -> (400, 100)
        mutant_genes = jax.tree_util.tree_map(flatten_batch, mutant_genes)
        
        return mutant_genes

    def _merge_and_evaluate(
        self, 
        key: chex.PRNGKey,
        elites_genes: chex.ArrayTree, 
        mutant_genes: chex.ArrayTree, 
        state: AbstractEvolutionState,
        params: GeneticEngineParams
    ) -> Tuple[BasePopulation, chex.Array]:
        """
        Combine elites and mutants, truncate to original size, and evaluate.
        """
        # Use state.population as the prototype for size and structure
        target_size = len(state.population)
        
        # 1. Determine sizes
        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        
        num_mutants_needed = target_size - num_elites
        
        # 2. Truncate mutants
        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants_needed], mutant_genes)
        
        # 3. Concatenate
        next_genes = jax.tree_util.tree_map(
            lambda e, m: jnp.concatenate([e, m], axis=0),
            elites_genes,
            mutants_keep
        )
        
        # 4. Wrap & Eval
        unevaluated_pop = state.population.replace(
            genes=next_genes,
            fitness=jnp.full((target_size,), -jnp.inf)
        )
        
        evaluated_pop = self.evaluator.evaluate_population(unevaluated_pop)
        fitness_values = evaluated_pop.fitness
        
        return evaluated_pop, fitness_values

    def _update_hall_of_fame(self, state: AbstractEvolutionState, new_pop: BasePopulation, params: GeneticEngineParams) -> Tuple[Any, float, int]:
        """
        Update global best genome, fitness, and stagnation counter.
        """
        best_idx = jnp.argmax(new_pop.fitness)
        curr_best_fit = new_pop.fitness[best_idx]
        
        # FIX: Extract .genes! 
        # new_pop[best_idx] is a Genome Object, state.best_genome is Raw Data.
        curr_best_genome = new_pop[best_idx].genes 

        # Opt direction correction for best fitness comparison
        # (Assuming maximization logic in fitness values)
        is_new_record = curr_best_fit > state.best_fitness
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new_record, new, old),
            state.best_genome,
            curr_best_genome
        )
        
        new_best_fit = jnp.maximum(state.best_fitness, curr_best_fit)
        new_stagnation = jnp.where(is_new_record, 0, state.stagnation_counter + 1)
        
        return new_best_genome, new_best_fit, new_stagnation

    def _execute_hooks(self, state: AbstractEvolutionState, params: GeneticEngineParams) -> AbstractEvolutionState:
        """Run all registered hooks."""
        new_state = state
        for hook in self.hooks:
            new_state = hook(new_state, params)
        return new_state

    # ==========================================
    # 2. INITIALIZATION & MAIN LOOP
    # ==========================================
    
    def init_state(self, rng_key: chex.Array, params: GeneticEngineParams) -> AbstractEvolutionState:
        """Initialize state from configuration."""
        init_pop_key, rng_key = jar.split(rng_key)
        
        # 1. Create initial population
        if isinstance(self.genome_config, BinaryGenomeConfig):
            population_class = BinaryPopulation
        elif isinstance(self.genome_config, RealGenomeConfig):
            population_class = RealPopulation
        elif isinstance(self.genome_config, CategoricalGenomeConfig):
            population_class = CategoricalPopulation
        else:
            raise ValueError(f"Unsupported genome config type: {type(self.genome_config)}")
        
        population = population_class.init_random(init_pop_key, self.genome_config, params.pop_size)

        # 2. Evaluate initial population
        evaluated_pop = self.evaluator.evaluate_population(population)
        fitness_values = evaluated_pop.fitness
        
        # 3. Find best
        # Optimization direction correction
        opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted_fitness = fitness_values * opt_sign
        
        best_idx = jnp.argmax(adjusted_fitness)
        best_genome = evaluated_pop[best_idx].genes 
        best_fit_val = fitness_values[best_idx]
        
        return AbstractEvolutionState(
            population=evaluated_pop,
            fitness_values=fitness_values,
            best_genome=best_genome,
            best_fitness=best_fit_val,
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
        Master Loop.
        """
        k_sel, k_gen, k_eval, k_next, k_misc = jar.split(key, 5)
        
        # 1. Elitism
        elites_genes = self._select_elites(k_misc, state, params)
        
        # 2. Selection
        parents = self._select_parents(k_sel, state, params)
        
        # 3. Variation
        mutants_genes = self._create_offspring(k_gen, parents, state, params)
        
        # 4. Merge & Eval
        new_pop, fitness_values = self._merge_and_evaluate(k_eval, elites_genes, mutants_genes, state, params)
        
        # 5. Update Statistics & State
        new_best_genome, new_best_fit, new_stagnation = self._update_hall_of_fame(state, new_pop, params)
        
        # 6. Create Intermediate State
        temp_state = state.replace(
            population=new_pop,
            fitness_values=fitness_values,
            best_genome=new_best_genome,
            best_fitness=new_best_fit,
            stagnation_counter=new_stagnation,
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        # 7. Run Hooks
        final_state = self._execute_hooks(temp_state, params)
        
        # 8. Runtime Logging
        if self.enable_progress_bar:
            jax.debug.callback(
                _host_progress_callback, 
                final_state.generation, 
                final_state.best_fitness
            )

        # 9. Return Metrics
        metrics = GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation
        )
        
        return k_next, final_state, metrics