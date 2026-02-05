"""
Standard Genetic Algorithm Engine.
Refactored for 'Init-Phase Compilation': Resource mapping happens once at initialization.
"""
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from ..core.base import BasePopulation
from ..core.fitness.base import BaseEvaluator
from ..core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from ..core.genome.categorical_genome import CategoricalGenomeConfig, CategoricalPopulation
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..operators.base import BaseCrossover, BaseMutation, BaseSelection
from .base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
)
from .resource_mapper import (
    KeyDerivationStrategy,
    ResourceMap,
    ShardingManager,
    compute_resource_map,
)

#TODO: update selection doctring

T = TypeVar("T", bound=Callable[..., Any])
_field = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

def traceable(name: str) -> Callable[[T], T]:
    """Correctly wraps a method in jax.named_call for HLO profiling labels."""
    def decorator(fn: T) -> T:
        # jax.named_call is untyped; cast it back to the original callable type
        return jax.named_call(fn, name=name)
    return decorator

@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """Configuration for Genetic Engine."""
    key_derivation: KeyDerivationStrategy = struct.field(pytree_node=False, default=KeyDerivationStrategy.SPLIT)  # type: ignore[no-untyped-call]
    mutation_strength_schedule: Optional[Callable[[int], float]] = struct.field( # type: ignore[no-untyped-call]
        pytree_node=False,
        default=None
    )

@struct.dataclass
class GeneticGenerationOutput(AbstractGenerationOutput):
    """KPIs returned by Genetic Engine."""
    random_key: chex.Array

@struct.dataclass
class OperatorState:
    """
    Holds the 'baked' operators with their static input sizes frozen.
    This ensures XLA compiles efficient kernels once.
    """
    selection: BaseSelection[Any, Any]
    crossover: BaseCrossover[Any, Any, Any]
    mutation: BaseMutation[Any, Any, Any]
@struct.dataclass
class GeneticEvolutionState(AbstractEvolutionState[BasePopulation[Any], Any]):
    """
    State that carries its own execution plan (ResourceMap) and optimized tools (OperatorState).
    """
    resource_map: ResourceMap = struct.field(pytree_node=False) # type: ignore[no-untyped-call]
    operators: OperatorState = struct.field(pytree_node=False) # type: ignore[no-untyped-call]

@struct.dataclass
class GeneticEngine(AbstractEngine[BasePopulation[Any], Any]):
    """
    The High-Performance Genetic Engine.
    
    Architecture: Init-Phase Compilation
    1. init_state: compiles ResourceMap & Bakes Operators.
    2. step: executes using pre-baked tools from State.
    """
    genome_config: Any = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    evaluator: BaseEvaluator[Any, Any, Any] = struct.field(pytree_node=False) # type: ignore[no-untyped-call]
    selection: BaseSelection[Any, Any] = struct.field(pytree_node=False) # type: ignore[no-untyped-call]
    crossover: BaseCrossover[Any, Any, Any] = struct.field(pytree_node=False) # type: ignore[no-untyped-call]
    mutation: BaseMutation[Any, Any, Any] = struct.field(pytree_node=False) # type: ignore[no-untyped-call]

    # Hooks & Config
    #hooks: Tuple[AbstractHook] = struct.field(default_factory=tuple)  # type: ignore[no-untyped-call]

    enable_progress_bar: bool = struct.field(pytree_node=False, default=False)  # type: ignore[no-untyped-call]

    _entropy_buffer: Tuple[Any, ...] = struct.field(pytree_node=False, default=())  # type: ignore[no-untyped-call]

    def __hash__(self) -> int:
        """Make engine hashable for JIT static_argnums."""
        return id(self)

    def __eq__(self,
               other: object
               ) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other

    @traceable("Phase_0_Allocate_Entropy")
    def _allocate_entropy(self, state: GeneticEvolutionState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        rmap = state.resource_map
        all_keys = rmap.get_keys(state.rng_key)

        k_sel_slice = all_keys[rmap.get_key_slice('selection')]
        k_cross = all_keys[rmap.get_key_slice('crossover')]
        k_mut   = all_keys[rmap.get_key_slice('mutation')]

        k_next  = all_keys[rmap.get_key_slice('next_key')][0]

        return k_sel_slice, k_cross, k_mut, k_next

    @traceable("Phase_0a_Get_Active_Operators")
    def _get_active_operators(self,
                              operators: OperatorState,
                              generation: int
                              ) -> OperatorState:
        """Returns OperatorState with scheduled mutation strength baked in."""
        if self.engine_params.mutation_strength_schedule is None:
            return operators

        scheduled_strength = self.engine_params.mutation_strength_schedule(generation)
        updated_mutation = cast(Any, operators.mutation).replace(mutation_strength=scheduled_strength)
        return cast(OperatorState, cast(Any, operators).replace(mutation=updated_mutation))

    @traceable("Phase_1_Selection_Read")
    def _selection_phase(self, key_selection: chex.Array, population: BasePopulation[Any], operators: OperatorState, params: AbstractEngineParams) -> Tuple[Any, chex.Array]:
        """
        Input: Specific key slice for selection.

        """
        # Handle zero-elitism safely: top_k with k=0 is invalid
        if params.elitism > 0:
            _, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
            elites_genes = population[elite_idx].genes
        else:
            # Create empty "genes" structure with 0 leading rows to preserve tree shape
            elites_genes = jax.tree_util.tree_map(lambda x: x[:0], population.genes)

        selected_idx = cast(chex.Array, operators.selection(key_selection, population))
        #parents = population[selected_idx]

        return elites_genes, selected_idx


    @traceable("Phase_2_Reproduction_Fused")
    def _reproduction_phase(
        self,
        keys_crossover: chex.Array,
        keys_mutation: chex.Array,
        parent_indices: chex.Array,
        population: BasePopulation[Any],
        operators: OperatorState,
        rmap: ResourceMap
    ) -> BasePopulation[Any]:

        num_pairs = rmap.crossover.input_count // 2
        p1_idx = parent_indices[:num_pairs]
        p2_idx = parent_indices[num_pairs : num_pairs * 2]

        # Debug assertions (dev-mode checks) 🔍
        # Parent indices should match expected input_count (2 * num_pairs)
        assert parent_indices.shape[0] == rmap.crossover.input_count, (
            f"Parent indices length mismatch: got {parent_indices.shape[0]}, expected {rmap.crossover.input_count}"
        )

        # Keys should be allocated per-pair (uses operator's num_keys contract)
        expected_cross_keys = operators.crossover.num_keys(input_shape=(num_pairs,))
        assert keys_crossover.shape[0] == expected_cross_keys, (
            f"Crossover keys length mismatch: got {keys_crossover.shape[0]}, expected {expected_cross_keys} (num_pairs={num_pairs})"
        )

        p1_pop = population[p1_idx]
        p2_pop = population[p2_idx]

        offspring_pop = cast(BasePopulation[Any], operators.crossover(  # type: ignore[call-arg]
            keys_crossover,
            p1_pop,
            p2_pop,
            self.genome_config
        ))

        # Validate that the crossover operator produced the expected number of offspring
        # This catches operators that report `num_offspring` but actually return a different
        # number of offspring in their `_recombine_one` implementation.
        produced_offspring = jax.tree_util.tree_leaves(offspring_pop.genes)[0].shape[0]
        assert produced_offspring == rmap.crossover.output_count, (
            f"Crossover produced {produced_offspring} offspring but ResourceMap expected {rmap.crossover.output_count}. "
            "Ensure `operator.num_offspring` matches the length of the tuple returned by `_recombine_one`."
        )

        final_pop = cast(BasePopulation[Any], operators.mutation(  # type: ignore[call-arg]
            keys_mutation,
            offspring_pop,
            self.genome_config
        ))

        return final_pop

    @traceable("Phase_3a_Merge")
    def _merge(self, elites_genes: Any, mutant_genes: Any, old_state: AbstractEvolutionState[BasePopulation[Any], Any]) -> Any:
        """Constructs the new population using the old_state shell."""
        target_size = len(old_state.population)

        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        num_mutants = target_size - num_elites

        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants], mutant_genes)

        next_genes = jax.tree_util.tree_map(
            lambda e, m: jnp.concatenate([e, m], axis=0),
            elites_genes, mutants_keep
        )

        return next_genes

    @traceable("Phase_3b_Evaluate")
    def _evaluate(self, new_genes: Any, old_state: AbstractEvolutionState[BasePopulation[Any], Any]) -> BasePopulation[Any]:
        new_population = cast(BasePopulation[Any], old_state.population.replace(genes=new_genes))
        evaluated_pop = cast(BasePopulation[Any], self.evaluator.evaluate_population(new_population))
        return evaluated_pop

    @traceable("Phase_3c_Update_HOF")
    def _update_hof(self, evaluated_pop: BasePopulation[Any], old_state: GeneticEvolutionState, k_next: chex.Array) -> GeneticEvolutionState:
        best_idx = jnp.argmax(evaluated_pop.fitness)
        curr_best_fit = evaluated_pop.fitness[best_idx]
        is_new = curr_best_fit > old_state.best_fitness

        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            old_state.best_genome, evaluated_pop[best_idx].genes
        )

        next_state = old_state.replace(
            population=evaluated_pop,
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, curr_best_fit, old_state.best_fitness),
            stagnation_counter=jnp.where(is_new, 0, old_state.stagnation_counter + 1),
            generation=old_state.generation + 1,
            rng_key=k_next
            # operators=old_state.operators (Implicitly preserved by replace)
            # resource_map=old_state.resource_map (Implicitly preserved)
        )
        return next_state

    @traceable("GeneticEngine_Step")
    def step(self, state: AbstractEvolutionState[BasePopulation[Any], Any]) -> Tuple[GeneticEvolutionState, GeneticGenerationOutput]:
        state = cast(GeneticEvolutionState, state)
        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        # Get operators with scheduled mutation strength baked in
        active_operators = self._get_active_operators(state.operators, state.generation)

        elites, parent_indices = self._selection_phase(k_sel, state.population, active_operators, self.engine_params)

        mutants = self._reproduction_phase(
                    k_cross,
                    k_mut,
                    parent_indices,
                    state.population,
                    active_operators,
                    state.resource_map
                )
        next_genes = self._merge(elites, mutants.genes, state)

        new_pop = self._evaluate(next_genes, state)

        final_state = self._update_hof(new_pop, state, k_next)

        # 7. HOOKS & METRICS
        #for hook in self.hooks:
        #   final_state = hook(final_state, self.engine_params)

        metrics = GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation,
            random_key=final_state.rng_key
        )

        '''if self.enable_progress_bar:
            jax.debug.callback(
                lambda g, f: print(f"Gen {g}: {f:.4f}"), 
                final_state.generation, final_state.best_fitness
            )'''

        return final_state, metrics


    def init_state(self, rng_key: chex.Array) -> GeneticEvolutionState:
        """
        Compiles the Execution Plan (ResourceMap), Bakes Operators, 
        and Enforces GSPMD Sharding Layout.
        """
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.engine_params.pop_size,
            self.engine_params.key_derivation
        )

        sharding_mgr = ShardingManager(axis_name='batch')
        active_sel = self.selection \
            .replace(num_selections=rmap.selection.output_count) \
            .set_input_length(rmap.selection.input_count)

        active_cross = self.crossover.set_input_length(rmap.crossover.input_count // 2)
        active_mut   = self.mutation.set_input_length(rmap.mutation.input_count)

        op_state = OperatorState(
            selection=active_sel,
            crossover=active_cross,
            mutation=active_mut
        )


        if isinstance(self.genome_config, BinaryGenomeConfig):
            pop_cls = BinaryPopulation
        elif isinstance(self.genome_config, RealGenomeConfig):
            pop_cls = RealPopulation
        elif isinstance(self.genome_config, CategoricalGenomeConfig):
            pop_cls = CategoricalPopulation
        else:
            raise ValueError(f"Unsupported config: {type(self.genome_config)}")

        init_pop_key, rng_key = jar.split(rng_key)


        population = pop_cls.init_random(
            init_pop_key,
            self.genome_config,
            self.engine_params.pop_size
        )

        target_dtype = self.genome_config.dtype

        def _enforce_layout(leaf: chex.Array) -> chex.Array:
            if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
                leaf = leaf.astype(target_dtype)

            if hasattr(leaf, 'shape') and len(leaf.shape) >= 2 and leaf.shape[0] == self.engine_params.pop_size:
                return jax.device_put(leaf, sharding_mgr.matrix_sharding)
            elif hasattr(leaf, 'shape') and len(leaf.shape) == 1 and leaf.shape[0] == self.engine_params.pop_size:
                return jax.device_put(leaf, sharding_mgr.vector_sharding)

            return jax.device_put(leaf, sharding_mgr.replicated_sharding)

        sharded_genes = jax.tree_util.tree_map(_enforce_layout, population.genes)
        fitness_casted = population.fitness.astype(target_dtype)
        sharded_fitness = jax.device_put(fitness_casted, sharding_mgr.vector_sharding)

        population = population.replace(genes=sharded_genes, fitness=sharded_fitness)


        evaluated_pop = self.evaluator.evaluate_population(population)
        best_idx = jnp.argmax(evaluated_pop.fitness)

        best_genome = jax.tree_util.tree_map(
            lambda x: jax.device_put(x[best_idx], sharding_mgr.replicated_sharding),
            evaluated_pop.genes
        )

        return GeneticEvolutionState(
            population=evaluated_pop,
            best_genome=best_genome,
            best_fitness=evaluated_pop.fitness[best_idx],
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0,
            resource_map=rmap,
            operators=op_state
        )

    # ==========================================
    # ASK / TELL Interface
    # ==========================================
    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """Allocate entropy for the next step and return population to evaluate.
        
        Returns:
            Tuple of (engine_with_entropy, population) - the engine carries the entropy buffer.
        """
        entropy = self._allocate_entropy(state)

        engine_with_entropy = cast(GeneticEngine, cast(Any, self).replace(_entropy_buffer=entropy))
        return engine_with_entropy, state.population

    def tell(self, state: GeneticEvolutionState, population: BasePopulation[Any]) -> GeneticEvolutionState:
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask().")

        k_sel, k_cross, k_mut, k_next = self._entropy_buffer

        state = state.replace(population=population)
        # HOF Update (Partial)
        best_idx = jnp.argmax(population.fitness)
        curr_best_fit = population.fitness[best_idx]
        is_new = curr_best_fit > state.best_fitness
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            state.best_genome, population[best_idx].genes
        )
        state = state.replace(
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, curr_best_fit, state.best_fitness),
            stagnation_counter=jnp.where(is_new, 0, state.stagnation_counter + 1)
        )
        elites, parent_indices = self._selection_phase(k_sel, state.population, state.operators, self.engine_params)

        # Get operators with scheduled mutation strength baked in
        active_operators = self._get_active_operators(state.operators, state.generation)

        mutants = self._reproduction_phase(
                    k_cross,
                    k_mut,
                    parent_indices,
                    state.population,
                    active_operators,
                    state.resource_map
                )

        next_genes = self._merge(elites, mutants.genes, state)
        next_population = cast(BasePopulation[Any], cast(Any, state.population).replace(genes=next_genes))

        final_state = cast(GeneticEvolutionState, cast(Any, state).replace(
            population=next_population,
            generation=state.generation + 1,
            rng_key=k_next
        ))

        #for hook in self.hooks:
        #    final_state = hook(final_state, self.engine_params)

        return final_state
