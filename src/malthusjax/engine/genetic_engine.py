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
from .inspector import inspect_engine_operators, EngineInspectionResult, ExecutionMode
from .resource_mapper import compute_resource_map, ResourceMap
from ..operators.base import BaseMutation, BaseCrossover, BaseSelection
from ..core.fitness.base import BaseEvaluator
from ..core.base import BasePopulation
from ..core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..core.genome.categorical_genome import CategoricalGenomeConfig, CategoricalPopulation


#TODO: Better Docstrings for all methods
#TODO: Add type hints for all methods
#TODO: slecting elites -> use get becaues select has a different meaning in selection operators

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
    
    Step 2 Enhancement: Inspector Integration
    The engine now automatically detects operator kernel support at
    initialization and sets execution mode (FAST_LANE vs LEGACY).
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
    
    @property
    def mode(self) -> ExecutionMode:
        """
        Get the current execution mode of the engine.
        
        Returns:
            ExecutionMode.FAST_LANE if all operators support kernel interface,
            ExecutionMode.LEGACY otherwise.
            
        Note:
            Inspection is performed on each access. For repeated access,
            use inspection_result property.
        """
        result = inspect_engine_operators(
            self.mutation, self.crossover, self.selection
        )
        return result.mode
    
    @property
    def inspection_result(self) -> EngineInspectionResult:
        """
        Get detailed inspection results for all operators.
        
        Returns:
            EngineInspectionResult with operator identity cards and mode.
            
        Note:
            Inspection is performed on each access since the engine is immutable.
        """
        return inspect_engine_operators(
            self.mutation, self.crossover, self.selection
        )
    
    @property
    def resource_map(self) -> ResourceMap:
        """
        Get the resource allocation map for this engine.
        
        Computes RNG budget and key slices for all operators based on
        genome configuration and population size. This enables static
        allocation and eliminates runtime RNG splitting overhead.
        
        Returns:
            ResourceMap with complete RNG budget allocation
            
        Note:
            Computed on each access. Cache externally if needed.
            
        Example:
            >>> engine = GeneticEngine(...)
            >>> rmap = engine.resource_map
            >>> print(f"Total RNG budget: {rmap.total_rng_budget}")
        """
        # Use a reasonable default pop_size if params not available
        # In practice, this will be called after init_state where pop_size is known
        default_pop_size = 100
        
        return compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            default_pop_size
        )
    
    # ==========================================
    # 1. MODULAR COMPONENT METHODS
    # ==========================================

    def _select_elites(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> chex.ArrayTree:
        """
        Preserve the top individuals.
        """
        # Respect optimization direction: convert fitness so that higher is better
        #opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted = state.population.fitness #* opt_sign
        _, elite_indices = jax.lax.top_k(adjusted, params.elitism)
        return state.population[elite_indices].genes

    def _select_parents(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> BasePopulation:
        """
        Select parents for reproduction.
        """
        # Pass adjusted fitness to selection so operators don't need to know
        # the optimization direction (maximize vs minimize).
        #opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted_fitness = state.population.fitness #* opt_sign
        indices = self.selection(key, adjusted_fitness)
    
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
        # Optimization direction correction
        #opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted_fitness = new_pop.fitness #* opt_sign
        
        best_idx = jnp.argmax(adjusted_fitness)  # Now works for both min and max
        curr_best_fit = new_pop.fitness[best_idx]
        
        curr_best_genome = new_pop[best_idx].genes 

        # Compare using adjusted fitness
        adjusted_state_best = state.best_fitness #* opt_sign
        adjusted_curr_best = curr_best_fit #* opt_sign
        is_new_record = adjusted_curr_best > adjusted_state_best
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new_record, new, old),
            state.best_genome,
            curr_best_genome
        )
        
        new_best_fit = jnp.where(is_new_record, curr_best_fit, state.best_fitness)
        new_stagnation = jnp.where(is_new_record, 0, state.stagnation_counter + 1)
        
        return new_best_genome, new_best_fit, new_stagnation

    def _execute_hooks(self, state: AbstractEvolutionState, params: GeneticEngineParams) -> AbstractEvolutionState:
        """Run all registered hooks."""
        new_state = state
        for hook in self.hooks:
            new_state = hook(new_state, params)
        return new_state

    # ==========================================
    # 2. EXECUTION PATHS (LEGACY vs FAST_LANE)
    # ==========================================
    
    def _step_legacy(
        self, 
        key: chex.Array, 
        state: AbstractEvolutionState, 
        params: GeneticEngineParams
    ) -> Tuple[chex.Array, AbstractEvolutionState, GeneticGenerationOutput]:
        """
        Legacy execution path (Python loop with RNG splitting per operator).
        Maintains backward compatibility and handles any operator configuration.
        
        Args:
            key: Master RNG key
            state: Current evolution state
            params: Engine parameters
            
        Returns:
            Tuple of (next_key, updated_state, metrics)
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

    def _step_fast(
        self,
        key: chex.Array,
        state: AbstractEvolutionState,
        params: GeneticEngineParams
    ) -> Tuple[chex.Array, AbstractEvolutionState, GeneticGenerationOutput]:
        """
        Fast-lane execution using pre-allocated RNG and buffer donation.
        
        Note: This is a foundational implementation that demonstrates
        the routing pattern and RNG pre-allocation strategy. Full vertical
        fusion (apply_kernel chaining) will be implemented in Step 5
        (operator migration) when operators provide fused kernels.
        
        For now, this path delegates to legacy logic but with pre-allocated
        RNG blocks to eliminate split overhead where operator kernels exist.
        
        Args:
            key: Master RNG key
            state: Current evolution state
            params: Engine parameters
            
        Returns:
            Tuple of (next_key, updated_state, metrics)
        """
        # Strict static allocation fast path using ResourceMap slices.
        pop_size = len(state.population)
        rmap = compute_resource_map(self.selection, self.crossover, self.mutation, self.genome_config, pop_size)

        # Helper to turn a slice into a keys array or empty array
        def _keys_for_slice(all_keys, sl):
            if all_keys is None:
                return jnp.empty((0, 2), dtype=jnp.uint32)
            if sl.start >= sl.stop:
                return jnp.empty((0, 2), dtype=all_keys.dtype)
            return all_keys[sl]

        # Check operator kernel support up-front. If any operator lacks kernel
        # support, fall back to the legacy path (do not run the strict fast scan).
        inspection = self.inspection_result
        if not (inspection.selection_card.supports_kernel and inspection.crossover_card.supports_kernel and inspection.mutation_card.supports_kernel):
            return self._step_legacy(key, state, params)

        # Inner scan body that will be JIT-compilable. Carry: (state, rng_key)
        def _fast_scan_body(carry, _):
            local_state, curr_key = carry
            population = local_state.population

            # Step 1: Master split (exactly one split per generation)
            next_key, gen_subkey = jar.split(curr_key)

            # Step 2: Entropy block allocation
            if rmap.total_rng_budget > 0:
                all_keys = jar.split(gen_subkey, rmap.total_rng_budget)
            else:
                all_keys = None

            # Step 3: Slicing (get key arrays for each operator)
            sel_slice = rmap.get_key_slice('selection')
            cross_slice = rmap.get_key_slice('crossover')
            mut_slice = rmap.get_key_slice('mutation')

            sel_keys = _keys_for_slice(all_keys, sel_slice)
            cross_keys = _keys_for_slice(all_keys, cross_slice)
            mut_keys = _keys_for_slice(all_keys, mut_slice)

            # Normalize single-key arrays to scalar keys when needed
            def _normalize(keys):
                if keys is None:
                    return None
                if keys.ndim == 2 and keys.shape[0] == 1:
                    return keys[0]
                return keys

            sel_keys = _normalize(sel_keys)
            cross_keys = _normalize(cross_keys)
            mut_keys = _normalize(mut_keys)

            # Step 4: Vertical fusion (Selection -> Crossover -> Mutation -> Eval)
            # If any operator lacks kernel support, fall back to legacy step
            inspection = self.inspection_result
            if not (inspection.selection_card.supports_kernel and inspection.crossover_card.supports_kernel and inspection.mutation_card.supports_kernel):
                # fallback: delegate to legacy path (keeps correctness)
                return (state.population, next_key), None

            # A. Elitism (deterministic)
            # Use the carried local_state (population-aware) for selection
            elites_genes = self._select_elites(next_key, local_state, params)

            # B. Selection kernel
            leaves, treedef = jax.tree_util.tree_flatten(population.genes)
            if len(leaves) == 0:
                # Use adjusted fitness so selection operators receive a
                # consistent "higher is better" signal
                #opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
                sel_idx = self.selection(next_key, population.fitness )#* opt_sign)
                parents = population[sel_idx]
            else:
                population_array = leaves[0]
                #opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
                sel_out = self.selection.apply_kernel(sel_keys, (population_array, population.fitness), params) #* opt_sign), params)
                # sel_out may be indices or array of rows
                if hasattr(sel_out, 'ndim') and sel_out.ndim == 1:
                    parents = population[sel_out]
                else:
                    new_genes = jax.tree_util.tree_unflatten(treedef, [sel_out])
                    parents = population.replace(genes=new_genes, fitness=jnp.full((sel_out.shape[0],), -jnp.inf))

            # C. Pairing: derive permutation key from crossover keys or fallback to gen_subkey
            def _pick_perm_key(karr):
                if karr is None:
                    return gen_subkey
                if getattr(karr, 'ndim', 0) == 1:
                    return karr
                if getattr(karr, 'ndim', 0) == 2 and karr.shape[0] > 0:
                    return karr[0]
                return gen_subkey

            perm_key = _pick_perm_key(cross_keys)
            p1 = parents
            p2_idx = jar.permutation(perm_key, jnp.arange(len(parents)))
            p2 = parents[p2_idx]

            # Extract primary arrays for kernels
            p1_arr = jax.tree_util.tree_leaves(p1.genes)[0]
            p2_arr = jax.tree_util.tree_leaves(p2.genes)[0]

            # D. Crossover kernel: expects per-pair keys (broadcast if needed)
            num_pairs = len(parents)
            if getattr(cross_keys, 'ndim', 0) == 2 and cross_keys.shape[0] >= num_pairs and num_pairs > 0:
                cross_keys_per_pair = cross_keys[:num_pairs]
            elif num_pairs > 0:
                # broadcast first key
                cross_keys_per_pair = jnp.stack([_pick_perm_key(cross_keys)] * num_pairs)
            else:
                cross_keys_per_pair = jnp.empty((0, 2), dtype=jnp.uint32)

            offspring_arr = self.crossover.apply_kernel(cross_keys_per_pair, (p1_arr, p2_arr), params)

            # E. Mutation kernel: expects per-offspring keys
            num_offspring = jax.tree_util.tree_leaves(offspring_arr)[0].shape[0]
            if getattr(mut_keys, 'ndim', 0) == 2 and mut_keys.shape[0] >= num_offspring and num_offspring > 0:
                mut_keys_per = mut_keys[:num_offspring]
            elif num_offspring > 0:
                mut_keys_per = jnp.stack([_pick_perm_key(mut_keys)] * num_offspring)
            else:
                mut_keys_per = jnp.empty((0, 2), dtype=jnp.uint32)

            mutated_arr = self.mutation.apply_kernel(mut_keys_per, offspring_arr, params)

            # F. Wrap mutated arrays into Genome structure matching population
            # Reuse population.replace to build unevaluated population
            new_genes = jax.tree_util.tree_unflatten(treedef, [mutated_arr])
            # Get population size from the first leaf (values array) shape
            pop_size = jax.tree_util.tree_leaves(new_genes)[0].shape[0]
            mutant_genes = new_genes  # Just the genes, not wrapped in population

            # G. Evaluation: choose an eval key (prefer remaining mutation keys)
            eval_key = _pick_perm_key(mut_keys)
            # Build a temporary state to call merge & evaluate
            temp_merge_state = local_state.replace(population=population)
            new_pop, fitness_values = self._merge_and_evaluate(eval_key, elites_genes, mutant_genes, temp_merge_state, params)

            # H. Update stats and state
            new_best_genome, new_best_fit, new_stagnation = self._update_hall_of_fame(local_state, new_pop, params)
            temp_state = local_state.replace(
                population=new_pop,
                fitness_values=fitness_values,
                best_genome=new_best_genome,
                best_fitness=new_best_fit,
                stagnation_counter=new_stagnation,
                generation=local_state.generation + 1,
                rng_key=next_key
            )

            final_state = self._execute_hooks(temp_state, params)

            metrics = GeneticGenerationOutput(
                best_fitness=final_state.best_fitness,
                mean_fitness=jnp.mean(new_pop.fitness),
                generation=final_state.generation
            )

            # Return the full updated state and the metrics for this generation
            return (final_state, next_key), metrics

        # Run the scan for a single generation step (carry -> new carry)
        (final_state, new_rng), metrics = jax.lax.scan(_fast_scan_body, (state, key), None, length=1)

        # If metrics is None, the scan indicated a fallback path; delegate to legacy
        if metrics is None:
            return self._step_legacy(key, state, params)

        # metrics is a single-element array-like structure produced by scan; unwrap
        gen_metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)

        # Return the next_key, final_state and generation metrics
        return new_rng, final_state, gen_metrics

    # ==========================================
    # 3. INITIALIZATION & MAIN LOOP
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
        # opt_sign= jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adjusted_fitness = fitness_values #* opt_sign
        
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
        Master router: selects LEGACY or FAST_LANE execution path based on engine mode.
        
        The engine automatically detects at runtime which operators support the
        apply_kernel interface and selects the appropriate execution path:
        
        - LEGACY mode: Python loop with per-operator RNG splitting (backward compatible)
        - FAST_LANE mode: JIT-compiled vertical fusion with pre-allocated RNG (optimized)
        
        User API is identical; the choice is transparent.
        
        Args:
            key: Master RNG key
            state: Current evolution state
            params: Engine parameters
            
        Returns:
            Tuple of (next_key, updated_state, metrics)
        """
        current_mode = self.mode
        
        # Route to appropriate execution path based on mode
        if current_mode == ExecutionMode.LEGACY:
            return self._step_legacy(key, state, params)
        else:
            # ExecutionMode.FAST_LANE
            return self._step_fast(key, state, params)