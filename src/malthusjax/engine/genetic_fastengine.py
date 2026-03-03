"""
Standard Genetic Algorithm Engine.
Refactored for 'Init-Phase Compilation': Resource mapping happens once at initialization.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.core.random import PRNGImpl, create_key, is_new_style_key, validate_key

from ..core.base import BaseGenome, BasePopulation
from ..core.fitness.base import BaseEvaluator
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
from .schedules import ScheduleType, compute_scheduled_strength

# TODO: update selection doctring

T = TypeVar("T", bound=Callable[..., Any])
_field: Any = struct.field  # Helper alias for typed dataclass fields


def traceable(name: str) -> Callable[[T], T]:
    """Correctly wraps a method in jax.named_call for HLO profiling labels."""

    def decorator(fn: T) -> T:
        # jax.named_call is untyped; cast it back to the original callable type
        return jax.named_call(fn, name=name)

    return decorator


@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """
    Configuration for Genetic Engine (extends base params).

    - key_derivation: KeyDerivationStrategy for RNG generation.
      SPLIT: Sequential jax.random.split (uncorrelated, single-threaded).
      FOLD: Parallel jax.random.fold_in (deterministic, parallelizable).
    - schedule_type: ScheduleType enum controlling mutation strength over
      generations.  ``CONSTANT`` (default) applies no schedule.
      ``LINEAR_DECAY``, ``COSINE_ANNEAL``, and ``EXPONENTIAL_DECAY`` are
      pure-JAX schedules safe inside ``jax.lax.scan``.
    - initial_strength: Mutation strength at generation 0 (used by
      non-CONSTANT schedules).  Defaults to ``0.1``.
    - final_strength: Target mutation strength at the last generation
      (used by LINEAR_DECAY and COSINE_ANNEAL).  Defaults to ``0.0``.
    - mutation_strength_schedule: **DEPRECATED** — Legacy Python callable.
      Use ``schedule_type``, ``initial_strength``, ``final_strength`` instead.
    """

    key_derivation: KeyDerivationStrategy = _field(
        pytree_node=False, default=KeyDerivationStrategy.SPLIT
    )
    prng_impl: PRNGImpl = _field(pytree_node=False, default=PRNGImpl.THREEFRY)
    schedule_type: ScheduleType = _field(
        pytree_node=False, default=ScheduleType.CONSTANT
    )
    initial_strength: float = 0.1
    final_strength: float = 0.0
    mutation_strength_schedule: Optional[Callable[[int], float]] = _field(
        pytree_node=False, default=None
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

    selection: BaseSelection[Any, Any] = _field(pytree_node=False)
    crossover: BaseCrossover[Any, Any, Any] = _field(pytree_node=False)
    mutation: BaseMutation[Any, Any, Any] = _field(pytree_node=False)


@struct.dataclass
class GeneticEvolutionState(AbstractEvolutionState[BaseGenome, BasePopulation[Any]]):
    """
    State that carries its own execution plan (ResourceMap) and optimized tools (OperatorState).
    """

    resource_map: ResourceMap = _field(pytree_node=False)
    operators: OperatorState = _field(pytree_node=False)


@struct.dataclass
class GeneticEngine(AbstractEngine[BaseGenome, BasePopulation[Any]]):
    """
    High-Performance Genetic Algorithm Engine (Init-Phase Compilation).
    Architecture: init_state() compiles ResourceMap, bakes operators with static input sizes,
    returns state carrying pre-baked tools. step() executes using cached resource plan.
    5-Phase per generation: (0) Entropy allocation → (1) Selection → (2) Reproduction →
    (3) Merge/Elite preservation → (4) Evaluate + HOF update.
    Ask/Tell interface: Alternative injection-style control flow (ask entropy, tell evaluated pop).
    """

    genome_config: Any = _field(pytree_node=False)
    evaluator: BaseEvaluator[Any, Any, Any] = _field(pytree_node=False)
    selection: BaseSelection[Any, Any] = _field(pytree_node=False)
    crossover: BaseCrossover[Any, Any, Any] = _field(pytree_node=False)
    mutation: BaseMutation[Any, Any, Any] = _field(pytree_node=False)
    # Hooks & Config
    # hooks: Tuple[AbstractHook] (placeholder for future hook support)

    enable_progress_bar: bool = _field(pytree_node=False, default=False)

    _entropy_buffer: Tuple[Any, ...] = _field(pytree_node=False, default=())

    def __hash__(self) -> int:
        """Make engine hashable for JIT static_argnums."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other

    @traceable("Phase_0_Allocate_Entropy")
    def _allocate_entropy(
        self, state: GeneticEvolutionState
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Allocate RNG keys for this generation (Phase 0).
        Slices pre-derived master key array into operator-specific subkeys.
        Returns: (k_selection, k_crossover, k_mutation, k_next_generation).
        Shapes: Each is (num_keys,) for that operator per resource_map allocation.
        """
        rmap = state.resource_map
        all_keys = rmap.get_keys(state.rng_key)

        k_sel_slice = all_keys[rmap.get_key_slice("selection")]
        k_cross = all_keys[rmap.get_key_slice("crossover")]
        k_mut = all_keys[rmap.get_key_slice("mutation")]

        k_next = all_keys[rmap.get_key_slice("next_key")][0]

        return k_sel_slice, k_cross, k_mut, k_next

    @traceable("Phase_0a_Get_Active_Operators")
    def _get_active_operators(self, operators: OperatorState, generation: int) -> OperatorState:
        """Returns OperatorState with scheduled mutation strength baked in.

        Uses the JAX-native ``ScheduleType`` enum when set.  Falls back to
        the **deprecated** ``mutation_strength_schedule`` callable for
        backward compatibility (emits a ``DeprecationWarning`` once).
        """
        params = cast(GeneticEngineParams, self.engine_params)

        # --- Legacy callable path (deprecated) ---------------------------------
        if params.mutation_strength_schedule is not None:
            warnings.warn(
                "mutation_strength_schedule is deprecated and will be removed in "
                "v0.4.0. Use schedule_type, initial_strength, and final_strength "
                "on GeneticEngineParams instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scheduled_strength = params.mutation_strength_schedule(generation)
            updated_mutation = cast(Any, operators.mutation).replace(
                mutation_strength=scheduled_strength
            )
            return cast(OperatorState, cast(Any, operators).replace(mutation=updated_mutation))

        # --- New JAX-native path ------------------------------------------------
        if params.schedule_type == ScheduleType.CONSTANT:
            return operators  # fast path: no struct mutation

        strength = compute_scheduled_strength(
            params.schedule_type,
            generation,
            params.num_generations,
            params.initial_strength,
            params.final_strength,
        )
        updated_mutation = cast(Any, operators.mutation).replace(
            mutation_strength=strength
        )
        return cast(OperatorState, cast(Any, operators).replace(mutation=updated_mutation))

    @traceable("Phase_1_Selection_Read")
    def _selection_phase(
        self,
        key_selection: chex.Array,
        population: BasePopulation[Any],
        operators: OperatorState,
        params: AbstractEngineParams,
    ) -> Tuple[Any, chex.Array]:
        """
        Select parents via elite preservation + selection operator (Phase 1).
        Elite handling: If elitism > 0, top_k extracts elite_idx, returns genes (0 leading rows).
        If elitism == 0, empty genes tree preserves structure for JAX tree_map.
        Selection output: selected_idx indices into population (shape: (num_selections,)).
        Returns: (elites_genes tree, selected_indices for mating).
        """
        # Handle zero-elitism safely: top_k with k=0 is invalid
        if params.elitism > 0:
            _, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
            elites_genes = population[elite_idx].genes
        else:
            # Create empty "genes" structure with 0 leading rows to preserve tree shape
            elites_genes = jax.tree_util.tree_map(lambda x: x[:0], population.genes)

        selected_idx = cast(chex.Array, operators.selection(key_selection, population))
        # parents = population[selected_idx]

        return elites_genes, selected_idx

    @traceable("Phase_2_Reproduction_Fused")
    def _reproduction_phase(
        self,
        keys_crossover: chex.Array,
        keys_mutation: chex.Array,
        parent_indices: chex.Array,
        population: BasePopulation[Any],
        operators: OperatorState,
        rmap: ResourceMap,
    ) -> BasePopulation[Any]:
        """
        Crossover + Mutation (Phase 2): Cascade: parents → offspring → mutants.
        Parent slicing: Split selected indices into (p1_idx, p2_idx) = first/second halves.
        Each pair (p1[i], p2[i]) produces num_offspring children via crossover.
        Offspring count may exceed pop_size (handled in merge phase).
        Returns: Mutated population (shape: (num_offspring * num_pairs, ...genome_shape)).
        """
        num_pairs = rmap.crossover.input_count // 2
        p1_idx = parent_indices[:num_pairs]
        p2_idx = parent_indices[num_pairs : num_pairs * 2]

        # Validate parent indices match expected input_count (2 * num_pairs)
        if parent_indices.shape[0] != rmap.crossover.input_count:
            raise ValueError(
                "Parent indices length mismatch: "
                f"got {parent_indices.shape[0]}, expected {rmap.crossover.input_count}"
            )

        # Keys should be allocated per-pair (uses operator's num_keys contract)
        expected_cross_keys = operators.crossover.num_keys(input_shape=(num_pairs,))
        if keys_crossover.shape[0] != expected_cross_keys:
            raise ValueError(
                "Crossover keys length mismatch: "
                f"got {keys_crossover.shape[0]}, expected {expected_cross_keys} (num_pairs={num_pairs})"
            )

        p1_pop = population[p1_idx]
        p2_pop = population[p2_idx]

        offspring_pop = cast(
            BasePopulation[Any],
            operators.crossover(keys_crossover, p1_pop, p2_pop, self.genome_config),
        )

        # Validate that the crossover operator produced the expected number of offspring
        # This catches operators that report `num_offspring` but actually return a different
        # number of offspring in their `_recombine_one` implementation.
        produced_offspring = jax.tree_util.tree_leaves(offspring_pop.genes)[0].shape[0]
        if produced_offspring != rmap.crossover.output_count:
            raise ValueError(
                f"Crossover produced {produced_offspring} offspring but ResourceMap "
                f"expected {rmap.crossover.output_count}. Ensure `operator.num_offspring` "
                "matches the length of the tuple returned by `_recombine_one`."
            )

        final_pop = cast(
            BasePopulation[Any],
            operators.mutation(keys_mutation, offspring_pop, self.genome_config),
        )

        return final_pop

    @traceable("Phase_3a_Merge")
    def _merge(
        self,
        elites_genes: Any,
        mutant_genes: Any,
        old_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]],
    ) -> Any:
        """
        Merge elite preservation + mutation results (Phase 3a).
        Strategy: Preserve top elites, fill remainder with mutant offspring.
        Slicing: num_elites from elites_genes + (pop_size - num_elites) from mutants.
        Returns: Concatenated genes tree ready for evaluation.
        """
        target_size = len(old_state.population)

        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        num_mutants = target_size - num_elites

        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants], mutant_genes)

        next_genes = jax.tree_util.tree_map(
            lambda e, m: jnp.concatenate([e, m], axis=0), elites_genes, mutants_keep
        )

        return next_genes

    @traceable("Phase_3b_Evaluate")
    def _evaluate(
        self, new_genes: Any, old_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]]
    ) -> BasePopulation[Any]:
        new_population = cast(Any, old_state.population).replace(genes=new_genes)
        evaluated_pop = self.evaluator.evaluate_population(new_population)
        return evaluated_pop

    @traceable("Phase_3c_Update_HOF")
    def _update_hof(
        self,
        evaluated_pop: BasePopulation[Any],
        old_state: GeneticEvolutionState,
        k_next: chex.Array,
    ) -> GeneticEvolutionState:
        best_idx = jnp.argmax(evaluated_pop.fitness)
        curr_best_fit = evaluated_pop.fitness[best_idx]
        is_new = curr_best_fit > old_state.best_fitness

        # Extract best genome by indexing genes directly (best_idx is a JAX array, not Python int)
        best_candidate = jax.tree_util.tree_map(lambda x: x[best_idx], evaluated_pop.genes)
        old_tree: Any = jax.tree_util.tree_structure(old_state.best_genome)
        cand_tree: Any = jax.tree_util.tree_structure(best_candidate)
        if old_tree != cand_tree:
            old_struct = jax.tree_util.tree_map(lambda _: old_state.best_genome, best_candidate)
        else:
            old_struct = old_state.best_genome

        new_best_genome = jax.lax.cond(
            is_new,
            lambda _: best_candidate,
            lambda _: old_struct,
            operand=None,
        )

        next_state = cast(
            GeneticEvolutionState,
            cast(Any, old_state).replace(
                population=evaluated_pop,
                best_genome=new_best_genome,
                best_fitness=jnp.where(is_new, curr_best_fit, old_state.best_fitness),
                stagnation_counter=jnp.where(is_new, 0, old_state.stagnation_counter + 1),
                generation=old_state.generation + 1,
                rng_key=k_next,
                # operators=old_state.operators (Implicitly preserved by replace)
                # resource_map=old_state.resource_map (Implicitly preserved)
            ),
        )
        return next_state

    @traceable("GeneticEngine_Step")
    def step(
        self, state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]]
    ) -> Tuple[GeneticEvolutionState, GeneticGenerationOutput]:
        state = cast(GeneticEvolutionState, state)
        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        # Get operators with scheduled mutation strength baked in
        active_operators = self._get_active_operators(state.operators, state.generation)

        elites, parent_indices = self._selection_phase(
            k_sel, state.population, active_operators, self.engine_params
        )

        mutants = self._reproduction_phase(
            k_cross, k_mut, parent_indices, state.population, active_operators, state.resource_map
        )
        next_genes = self._merge(elites, mutants.genes, state)

        new_pop = self._evaluate(next_genes, state)

        final_state = self._update_hof(new_pop, state, k_next)

        # 7. HOOKS & METRICS
        # for hook in self.hooks:
        #   final_state = hook(final_state, self.engine_params)

        metrics = GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation,
            random_key=final_state.rng_key,
        )

        """if self.enable_progress_bar:
            jax.debug.callback(
                lambda g, f: print(f"Gen {g}: {f:.4f}"),
                final_state.generation, final_state.best_fitness
            )"""

        return final_state, metrics

    def init_state(self, rng_key: Union[int, chex.Array]) -> GeneticEvolutionState:
        """
        Initialize evolution state (Init-Phase Compilation).

        Accepts either an integer seed (convenience) or a pre-constructed PRNG key.
        If an integer seed is provided, a typed key is created using the engine's
        configured `prng_impl`. If a legacy `PRNGKey` is provided, a
        ``DeprecationWarning`` is emitted.

        Steps: (1) Compute ResourceMap (RNG budget + data flow cascade).
        (2) Bake operators: set_input_length freezes static sizes for XLA.
        (3) Enforce GSPMD sharding layout (per-device or replicated).
        (4) Initialize and evaluate population.
        (5) Return state carrying resource_map + operators for all steps().
        One-time cost; results cached in state.resource_map throughout run.
        """
        params = cast(GeneticEngineParams, self.engine_params)

        # Accept an integer seed and create a typed key using the configured PRNG impl,
        # otherwise validate provided key and warn for legacy PRNGKey.
        if isinstance(rng_key, int):
            rng_key = create_key(rng_key, impl=params.prng_impl)
        else:
            validate_key(rng_key, context="GeneticEngine.init_state()")

        # Determine key format once — propagated to all operators as static flag.
        typed = is_new_style_key(rng_key)

        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.engine_params.pop_size,
            params.key_derivation,
        )

        sharding_mgr = ShardingManager(axis_name="batch")
        active_sel = (
            cast(Any, self.selection)
            .replace(num_selections=rmap.selection.output_count)
            .set_input_length(rmap.selection.input_count)
            .set_typed_keys(typed)
        )

        # configure crossover with correct input length and key type
        active_cross = self.crossover.set_input_length(
            rmap.crossover.input_count // 2
        ).set_typed_keys(typed)
        active_mut = self.mutation.set_input_length(rmap.mutation.input_count).set_typed_keys(typed)

        op_state = OperatorState(selection=active_sel, crossover=active_cross, mutation=active_mut)

        pop_cls: Type[BasePopulation[Any]]
        cfg: Any
        init_pop_key, rng_key = jar.split(rng_key)

        # Protocol dispatch: config.init_population() replaces isinstance chain (JR-2)
        if not hasattr(self.genome_config, 'init_population'):
            raise ValueError(
                f"Unsupported genome config: {type(self.genome_config).__name__}. "
                "Config must implement init_population(key, size) -> BasePopulation."
            )
        population = self.genome_config.init_population(init_pop_key, self.engine_params.pop_size)

        target_dtype = self.genome_config.dtype

        def _enforce_layout(leaf: chex.Array) -> chex.Array:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                leaf = leaf.astype(target_dtype)

            if (
                hasattr(leaf, "shape")
                and len(leaf.shape) >= 2
                and leaf.shape[0] == self.engine_params.pop_size
            ):
                return jax.device_put(leaf, sharding_mgr.matrix_sharding)
            elif (
                hasattr(leaf, "shape")
                and len(leaf.shape) == 1
                and leaf.shape[0] == self.engine_params.pop_size
            ):
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
            evaluated_pop.genes,
        )

        return GeneticEvolutionState(
            population=evaluated_pop,
            best_genome=best_genome,
            best_fitness=evaluated_pop.fitness[best_idx],
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0,
            resource_map=rmap,
            operators=op_state,
        )

    # ==========================================
    # ASK / TELL Interface
    # ==========================================
    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """
        Ask for next evaluation batch (injection-style interface).
        Allocates entropy for this generation, returns (engine_with_entropy, population).
        Call pattern: engine, pop = ask(state); evaluated_pop = evaluate(pop);
        new_state = tell(state, evaluated_pop).
        Alternative to step() for external fitness evaluators.
        """
        entropy = self._allocate_entropy(state)

        engine_with_entropy = cast(GeneticEngine, cast(Any, self).replace(_entropy_buffer=entropy))
        return engine_with_entropy, state.population

    def tell(
        self, state: GeneticEvolutionState, population: BasePopulation[Any]
    ) -> GeneticEvolutionState:
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask().")

        k_sel, k_cross, k_mut, k_next = self._entropy_buffer

        state = cast(GeneticEvolutionState, cast(Any, state).replace(population=population))
        # HOF Update (Partial)
        best_idx = jnp.argmax(population.fitness)
        curr_best_fit = population.fitness[best_idx]
        is_new = curr_best_fit > state.best_fitness
        # Extract best candidate using tree_map in case best_idx is a JAX array
        best_candidate = jax.tree_util.tree_map(lambda x: x[best_idx], population.genes)
        # Ensure both branches return the same pytree structure.
        if jax.tree_util.tree_structure(state.best_genome) != jax.tree_util.tree_structure(
            best_candidate
        ):
            state_struct = jax.tree_util.tree_map(lambda _: state.best_genome, best_candidate)
        else:
            state_struct = state.best_genome

        new_best_genome = jax.lax.cond(
            is_new,
            lambda _: best_candidate,
            lambda _: state_struct,
            operand=None,
        )
        state = cast(
            GeneticEvolutionState,
            cast(Any, state).replace(
                best_genome=new_best_genome,
                best_fitness=jnp.where(is_new, curr_best_fit, state.best_fitness),
                stagnation_counter=jnp.where(is_new, 0, state.stagnation_counter + 1),
            ),
        )
        elites, parent_indices = self._selection_phase(
            k_sel, state.population, state.operators, self.engine_params
        )

        # Get operators with scheduled mutation strength baked in
        active_operators = self._get_active_operators(state.operators, state.generation)

        mutants = self._reproduction_phase(
            k_cross, k_mut, parent_indices, state.population, active_operators, state.resource_map
        )

        next_genes = self._merge(elites, mutants.genes, state)
        next_population = cast(
            BasePopulation[Any], cast(Any, state.population).replace(genes=next_genes)
        )

        final_state = cast(
            GeneticEvolutionState,
            cast(Any, state).replace(
                population=next_population, generation=state.generation + 1, rng_key=k_next
            ),
        )

        # for hook in self.hooks:
        #    final_state = hook(final_state, self.engine_params)

        return final_state
