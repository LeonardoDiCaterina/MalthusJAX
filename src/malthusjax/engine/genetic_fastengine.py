"""
Standard Genetic Algorithm Engine.
Refactored for 'Init-Phase Compilation': Resource mapping happens once at initialization.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, TypeVar, Union, cast

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
from .schedules import ScheduleType, TrackBest

# TODO: update selection docstring with expected input/output contract details.

T = TypeVar("T", bound=Callable[..., Any])
_field: Any = struct.field  # Helper alias for typed dataclass fields

# ---------------------------------------------------------------------------
# HLO tracing gate
# ---------------------------------------------------------------------------
# By default tracing is OFF so XLA can fuse all 5 phases into a single kernel.
# Set ``debug_tracing=True`` in GeneticEngineParams (or call enable_tracing())
# to re-enable jax.named_call labels for profiling / HLO inspection.
# ---------------------------------------------------------------------------
_TRACING_ENABLED: bool = False


def enable_tracing() -> None:
    """Enable jax.named_call phase labels globally (for HLO profiling)."""
    global _TRACING_ENABLED
    _TRACING_ENABLED = True


def disable_tracing() -> None:
    """Disable jax.named_call phase labels (default; allows XLA kernel fusion)."""
    global _TRACING_ENABLED
    _TRACING_ENABLED = False


def traceable(name: str) -> Callable[[T], T]:
    """Wraps a method in jax.named_call when _TRACING_ENABLED is True.

    Both the raw function and the named variant are pre-built at decoration
    time.  The ``if _TRACING_ENABLED`` check is a pure Python branch evaluated
    once per JAX trace, so there is zero XLA overhead when tracing is off.
    """

    def decorator(fn: T) -> T:
        named_fn = jax.named_call(fn, name=name)

        def conditional(*args: Any, **kwargs: Any) -> Any:
            if _TRACING_ENABLED:
                return named_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        return cast(T, conditional)

    return decorator


@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """Configuration for Genetic Algorithm Engine (extends AbstractEngineParams).

    This dataclass holds all static configuration for the genetic engine.
    All fields use ``pytree_node=False`` to trigger recompilation when changed.

    Inherited from AbstractEngineParams
    -----------------------------------
    pop_size : int
        Number of individuals in population. Must be > 0.
        **GPU efficiency**: Powers of 2 (32, 64, 128, 256) strongly preferred.
        Boundary: pop_size <= 0 raises ValueError in validate_engine_params().

    elitism : int
        Number of best individuals copied unchanged to next generation.
        Valid range: 0 <= elitism < pop_size.
        - 0: No elitism (all individuals selected/bred)
        - 1-N: Preserve best-N (typical: 1-5% of pop_size)
        Boundary: elitism >= pop_size raises ValueError in validate_engine_params().

    num_generations : int
        Total generational cycles to run. Must be > 0.
        Baked into JIT-compiled code; changing triggers recompilation.

    Additional Configuration
    ========================
    key_derivation : KeyDerivationStrategy, optional
        RNG splitting strategy for entropy allocation.
        - SPLIT: Sequential jax.random.split (uncorrelated, traditional)
        - FOLD: Parallel jax.random.fold_in (deterministic, vectorizable)
        Default: KeyDerivationStrategy.SPLIT.

    prng_impl : PRNGImpl, optional
        PRNG type for converting integer seeds to typed keys.
        - THREEFRY: Default, deterministic on all platforms
        - PHILOX: Fast on GPU, may vary cross-platform
        Default: PRNGImpl.THREEFRY.

    schedule_type : ScheduleType, optional
        Mutation strength schedule across generations.
        - CONSTANT: No schedule (apply static strength throughout)
        - LINEAR_DECAY: Linearly fade from initial_strength to final_strength
        - COSINE_ANNEAL: Cosine annealing (smooth decay)
        - EXPONENTIAL_DECAY: Exponential decay
        Default: ScheduleType.CONSTANT.

    initial_strength : float, optional
        Mutation strength at generation 0 (used by non-CONSTANT schedules).
        Typical range: [0.05, 0.5]. Default: 0.1.

    final_strength : float, optional
        Mutation strength at generation = num_generations - 1.
        Typical range: [0.0, 0.1]. Default: 0.0.

    track_best : TrackBest, optional
        Hall-of-Fame (best individual) tracking strategy.
        - NONE: No tracking; compute best_genome once at end (fastest)
        - LIGHT: Track best_fitness as running max, but not genome
        - FULL: Track both best_fitness and best_genome every generation (slowest)
        Default: TrackBest.LIGHT.

    debug_tracing : bool, optional
        Enable jax.named_call labels for XLA HLO profiling/inspection.
        When False (default), allows XLA to fuse all 5 phases into one kernel.
        Set True to see phase-level time breakdowns in profilers.
        Default: False.

    Notes
    -----
    Call :func:`validate_engine_params` before starting evolution to catch
    configuration errors (e.g., invalid elitism) outside JIT context.

    All fields are ``pytree_node=False``, so ``.replace()`` creates a new
    instance that will trigger XLA recompilation on next execution.
    """

    key_derivation: KeyDerivationStrategy = _field(
        pytree_node=False, default=KeyDerivationStrategy.SPLIT
    )
    prng_impl: PRNGImpl = _field(pytree_node=False, default=PRNGImpl.THREEFRY)
    schedule_type: ScheduleType = _field(pytree_node=False, default=ScheduleType.CONSTANT)
    track_best: TrackBest = _field(pytree_node=False, default=TrackBest.LIGHT)
    initial_strength: float = 0.1
    final_strength: float = 0.0
    debug_tracing: bool = _field(pytree_node=False, default=False)
    """Enable jax.named_call phase labels for HLO profiling (default: False).

    When False (default) the traceable decorators are no-ops, allowing XLA to
    fuse all phases into a single kernel.  Set to True to get named labels in
    the XLA HLO / profiler output.
    """


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
    """High-Performance Genetic Algorithm Engine with Init-Phase JIT Compilation.

    **Key Design**: The engine compiles once during :meth:`init_state`, caching the
    complete execution plan (ResourceMap) and optimized operators. Subsequent calls
    to :meth:`step` use the cached plan, avoiding recompilation overhead.

    **Architecture**: 5 phases per generation—

    1. **Entropy Allocation**: Master PRNG key split into subkeys for each operator
    2. **Selection**: Choose parent indices via tournament, roulette, or elite pool
    3. **Reproduction**: Crossover selected pairs + mutate offspring
    4. **Merge**: Combine elites (if elitism > 0) with mutants; preserve population size
    5. **Evaluation**: Score new genomes; update Hall-of-Fame (best individual tracking)

    **Interfaces**:

    - **Scan-based** (:meth:`run`): Calls :meth:`step` in jax.lax.scan loop. Preferred for
      internal evaluation and full reproducibility.
    - **Ask/Tell** (:meth:`ask`, :meth:`tell`): Entropy-first interface. Call :meth:`ask` to
      get entropy, evaluate externally, then call :meth:`tell` with results. Ideal for
      custom fitness evaluators or distributed evaluation.

    **JIT Compilation Behavior**: :meth:`init_state` triggers one-time XLA compilation
    that bakes operator input sizes and sharding layout. All subsequent :meth:`step`
    calls use cached compiled kernels. Changing engine_params.num_generations or
    operator specs requires re-calling :meth:`init_state` (not automatic).

    Parameters
    ----------
    genome_config : Any
        Genome configuration specifying the genome type, bounds, and initialization.
        Must implement ``init_population(key: Array, size: int) -> BasePopulation``
        and expose a ``dtype`` property for type conversions.
        Typically a :class:`~malthusjax.core.genome.RealGenomeConfig` or
        :class:`~malthusjax.core.genome.BinaryGenomeConfig`.

    evaluator : BaseEvaluator
        Fitness evaluator function. Must implement ``evaluate_population(pop: BasePopulation)``
        returning a :class:`BasePopulation` with fitness scores filled in.
        See :class:`~malthusjax.core.fitness.base.BaseEvaluator`.

    selection : BaseSelection
        Selection operator for choosing parents from population.
        Must implement ``__call__(key, fitness) -> (parent_idx, elite_idx)``.
        Examples: :class:`~malthusjax.operators.selection.TournamentSelection`,
        :class:`~malthusjax.operators.selection.RouletteSelection`,
        :class:`~malthusjax.operators.selection.ElitePoolSelection`.

    crossover : BaseCrossover
        Crossover operator for combining parent genomes.
        Must implement ``__call__(key, parent1_pop, parent2_pop, genome_config, generation)
        -> offspring_pop``.
        Examples: :class:`~malthusjax.operators.crossover.UniformRealCrossover`,
        :class:`~malthusjax.operators.crossover.BlendCrossover`,
        :class:`~malthusjax.operators.crossover.SimulatedBinaryCrossover`,
        :class:`~malthusjax.operators.crossover.UniformBinaryCrossover`,
        :class:`~malthusjax.operators.crossover.SinglePointCrossover`.

    mutation : BaseMutation
        Mutation operator for perturbing offspring.
        Must implement ``__call__(key, population, genome_config, generation) -> mutated_pop``.
        Examples: :class:`~malthusjax.operators.mutation.GaussianMutation`,
        :class:`~malthusjax.operators.mutation.PolynomialMutation`,
        :class:`~malthusjax.operators.mutation.BitflipMutation`,
        :class:`~malthusjax.operators.mutation.BallMutation`,
        :class:`~malthusjax.operators.mutation.ScrambleMutation`.

    engine_params : GeneticEngineParams
        Static configuration for the engine (population size, elitism, generations, etc.).
        See :class:`GeneticEngineParams` for all tunable parameters.

    enable_progress_bar : bool, optional
        If True, display progress bar during :meth:`run` (requires ``tqdm``).
        Default: False.

    Notes
    -----
    **Immutability**: This is a flax.struct dataclass, all fields are keyword-only.
    Create instances with ``GeneticEngine(genome_config=..., evaluator=..., ...)``.

    **Static Fields**: ``genome_config``, ``evaluator``, and operators are marked
    ``pytree_node=False``, so they don't participate in JAX tree operations. Changing
    them requires re-calling :meth:`init_state`.
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
        """Phase 0 — slice the master RNG buffer into operator subkeys.

        The state contains a precomputed ``ResourceMap``. This method pulls the
        right slices for selection, crossover, mutation and the next-generation
        key, returning them as a quadruple of key arrays.
        """
        rmap = state.resource_map
        all_keys = rmap.get_keys(state.rng_key)

        k_sel_slice = all_keys[rmap.get_key_slice("selection")]
        k_cross = all_keys[rmap.get_key_slice("crossover")]
        k_mut = all_keys[rmap.get_key_slice("mutation")]

        k_next = all_keys[rmap.get_key_slice("next_key")][0]

        return k_sel_slice, k_cross, k_mut, k_next

    @traceable("Phase_1_Selection_Read")
    def _selection_phase(
        self,
        key_selection: chex.Array,
        population: BasePopulation[Any],
        operators: OperatorState,
        params: AbstractEngineParams,
    ) -> Tuple[Any, chex.Array]:
        """Phase 1 — run selection and optionally gather elites.

        Uses the fused ``operators.selection`` call which may yield both parent
        and elite indices in one pass. The method then optionally slices the
        current population to extract elite genomes for preservation.
        """
        parent_idx, elite_idx = operators.selection(key_selection, population.fitness)

        if params.elitism > 0:
            elites_genes = jax.tree_util.tree_map(lambda x: x[elite_idx], population.genes)
        else:
            elites_genes = jax.tree_util.tree_map(lambda x: x[:0], population.genes)

        return elites_genes, parent_idx

    @traceable("Phase_2_Reproduction_Fused")
    def _reproduction_phase(
        self,
        keys_crossover: chex.Array,
        keys_mutation: chex.Array,
        parent_indices: chex.Array,
        population: BasePopulation[Any],
        operators: OperatorState,
        rmap: ResourceMap,
        generation: int = 0,
    ) -> BasePopulation[Any]:
        """Phase 2 — perform crossover followed by mutation.

        The incoming parent indices are split into two equal halves representing
        mating pairs. Crosser and mutator are then invoked with their
        respective key bundles, producing an intermediate offspring population
        which is returned for merging.
        """
        num_pairs = rmap.crossover.input_count // 2
        p1_idx = parent_indices[:num_pairs]
        p2_idx = parent_indices[num_pairs : num_pairs * 2]

        if parent_indices.shape[0] != rmap.crossover.input_count:
            raise ValueError(
                "Parent indices length mismatch: "
                f"got {parent_indices.shape[0]}, expected {rmap.crossover.input_count}"
            )

        expected_cross_keys = operators.crossover.num_keys(input_shape=(num_pairs,))
        if keys_crossover.shape[0] != expected_cross_keys:
            raise ValueError(
                "Crossover keys length mismatch: "
                f"got {keys_crossover.shape[0]}, expected {expected_cross_keys} "
                f"(num_pairs={num_pairs})"
            )
        p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
        p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
        dummy_fitness = jnp.zeros(num_pairs)
        p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
        p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

        offspring_pop = cast(
            BasePopulation[Any],
            operators.crossover(
                keys_crossover, p1_pop, p2_pop, self.genome_config, generation=generation
            ),
        )
        produced_offspring = jax.tree_util.tree_leaves(offspring_pop.genes)[0].shape[0]
        if produced_offspring != rmap.crossover.output_count:
            raise ValueError(
                f"Crossover produced {produced_offspring} offspring but ResourceMap "
                f"expected {rmap.crossover.output_count}. Ensure `operator.num_offspring` "
                f"matches the length of the tuple returned by `_recombine_one`."
            )

        final_pop = cast(
            BasePopulation[Any],
            operators.mutation(
                keys_mutation, offspring_pop, self.genome_config, generation=generation
            ),
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

        Uses ``jax.lax.dynamic_update_slice`` instead of ``jnp.concatenate``
        (FB-3) so that the output buffer can be donated by XLA — the
        concatenate op always allocates a new buffer, defeating donation.

        Strategy: Pre-allocate output via ``jnp.empty_like``, write elites
        into rows ``[0, num_elites)``, then mutants into
        ``[num_elites, pop_size)``.
        """
        target_size = len(old_state.population)

        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        num_mutants = target_size - num_elites

        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants], mutant_genes)

        def _fuse(old: jnp.ndarray, elite: jnp.ndarray, mutant: jnp.ndarray) -> jnp.ndarray:
            buf = old
            if num_elites > 0:
                start = tuple([0] * len(buf.shape))
                buf = jax.lax.dynamic_update_slice(buf, elite, start)
            mutant_start = tuple([num_elites] + [0] * (len(buf.shape) - 1))
            buf = jax.lax.dynamic_update_slice(buf, mutant, mutant_start)
            return buf

        old_genes = old_state.population.genes
        next_genes = jax.tree_util.tree_map(_fuse, old_genes, elites_genes, mutants_keep)

        return next_genes

    @traceable("Phase_3b_Evaluate")
    def _evaluate(
        self, new_genes: Any, old_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]]
    ) -> BasePopulation[Any]:
        new_population = cast(Any, old_state.population).replace(genes=new_genes)
        evaluated_pop = self.evaluator.evaluate_population(new_population)
        return evaluated_pop

    @traceable("GeneticEngine_Step")
    def step(
        self, state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]]
    ) -> Tuple[GeneticEvolutionState, GeneticGenerationOutput]:
        state = cast(GeneticEvolutionState, state)
        params = cast(GeneticEngineParams, self.engine_params)

        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        elites, parent_indices = self._selection_phase(
            k_sel, state.population, state.operators, self.engine_params
        )

        mutants = self._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            state.population,
            state.operators,
            state.resource_map,
            generation=state.generation,
        )
        next_genes = self._merge(elites, mutants.genes, state)

        new_pop = self._evaluate(next_genes, state)

        # ------------------------------------------------------------------
        # Inline HOF update (replaces _update_hof — FB-4)
        #
        # Python-level branching on ``track_best`` is safe because it is a
        # ``pytree_node=False`` field — only one branch is ever traced.
        # Evaluators now use minimization convention (lower=better), so the
        # engine uses jnp.min / jnp.minimum / jnp.argmin uniformly.
        # ------------------------------------------------------------------
        gen_best_fitness = jnp.min(new_pop.fitness)

        if params.track_best == TrackBest.NONE:
            """No tracking: pass through unchanged, report per-gen best"""
            new_best_fitness = state.best_fitness
            new_best_genome = state.best_genome
            metric_best = gen_best_fitness  # per-gen, NOT monotonic
        elif params.track_best == TrackBest.LIGHT:
            """Running min only — no genome in carry """
            new_best_fitness = jnp.minimum(gen_best_fitness, state.best_fitness)
            new_best_genome = state.best_genome
            metric_best = new_best_fitness  # monotonic
        else:
            """Full tracking: argmin + Gather + element-wise jnp.where """
            is_new = gen_best_fitness < state.best_fitness
            best_idx = jnp.argmin(new_pop.fitness)
            new_best_fitness = jnp.where(is_new, gen_best_fitness, state.best_fitness)
            best_candidate = jax.tree_util.tree_map(lambda x: x[best_idx], new_pop.genes)
            new_best_genome = jax.tree_util.tree_map(
                lambda n, o: jnp.where(is_new, n, o),
                best_candidate,
                state.best_genome,
            )
            metric_best = new_best_fitness  # monotonic

        final_state = cast(
            GeneticEvolutionState,
            cast(Any, state).replace(
                population=new_pop,
                best_genome=new_best_genome,
                best_fitness=new_best_fitness,
                generation=state.generation + 1,
                rng_key=k_next,
            ),
        )

        metrics = GeneticGenerationOutput(
            best_fitness=metric_best,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation,
            random_key=final_state.rng_key,
        )

        return final_state, metrics

    def run(
        self,
        initial_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
    ) -> Tuple[
        AbstractEvolutionState[BaseGenome, BasePopulation[Any]], AbstractGenerationOutput, Any
    ]:
        """Execute full evolution from initial state using jax.lax.scan.

        This method runs the genetic algorithm for num_generations by repeatedly
        calling :meth:`step` inside a :func:`jax.lax.scan` loop. It handles both
        internal fitness evaluation and Hall-of-Fame tracking.

        Parameters
        ----------
        initial_state : GeneticEvolutionState
            Starting state (typically from :meth:`init_state`). Must contain
            population, resource_map, and baked operators.

        time_it : bool, optional
            If True, measure wall-clock time for the entire scan loop (Python-side).
            Default: False. Note: timing includes XLA compilation on first call.

        compile : bool, optional
            Deprecated. For compatibility only; has no effect.
            Default: True.

        verbose : bool, optional
            If True, print progress bar (requires tqdm). Default: False.

        Returns
        -------
        final_state : GeneticEvolutionState
            State after num_generations steps. Contains population, best_genome,
            best_fitness, and current generation.

        history : List[GeneticGenerationOutput]
            Per-generation KPIs. Each item contains:

            - ``best_fitness`` : Best fitness this generation
            - ``mean_fitness`` : Population mean fitness
            - ``generation`` : Generation counter
            - ``random_key`` : PRNG key for reproducibility

        elapsed_time : float or None
            Wall-clock time in seconds (only if time_it=True, else None).

        Notes
        -----
        **Fitness Evaluation**: The engine calls :meth:`evaluator.evaluate_population`
        internally after each generation's reproduction phase.

        **Hall-of-Fame Tracking**: Best individual tracking strategy
        (from :attr:`engine_params.track_best`)—

        - NONE: No per-generation tracking; best_genome computed once at end
        - LIGHT: best_fitness tracked as running max each generation
        - FULL: Both best_fitness and best_genome updated every generation

        **Compilation**: First call triggers XLA compilation (may take seconds for
        large populations). Subsequent calls use cached compiled kernels.

        Examples
        --------
        Run a full evolution::

            state = engine.init_state(rng_key=42)
            final_state, history, elapsed = engine.run(state, time_it=True)
            print(f"Evolution took {elapsed:.2f} seconds")
            print(f"Best fitness: {final_state.best_fitness}")

        Extract convergence history::

            best_per_gen = [h.best_fitness for h in history]
            import matplotlib.pyplot as plt
            plt.plot(best_per_gen)
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.show()
        """
        final_state, history, elapsed_time = super().run(
            initial_state, time_it=time_it, compile=compile, verbose=verbose
        )
        params = cast(GeneticEngineParams, self.engine_params)
        if params.track_best in (TrackBest.NONE, TrackBest.LIGHT):
            best_idx = jnp.argmax(final_state.population.fitness)
            final_best_genome = jax.tree_util.tree_map(
                lambda x: x[best_idx], final_state.population.genes
            )
            final_state = cast(
                GeneticEvolutionState,
                cast(Any, final_state).replace(best_genome=final_best_genome),
            )

        if params.track_best == TrackBest.NONE:
            best_idx = jnp.argmax(final_state.population.fitness)
            final_state = cast(
                GeneticEvolutionState,
                cast(Any, final_state).replace(
                    best_fitness=final_state.population.fitness[best_idx]
                ),
            )

        return final_state, history, elapsed_time

    def init_state(self, rng_key: Union[int, chex.Array]) -> GeneticEvolutionState:
        """Initialize evolution and compile the inference plan (Init-Phase Compilation).

        This method performs the expensive one-time setup that all :meth:`step` calls
        will rely on. It:

        1. **Computes ResourceMap**: Pre-calculates RNG budget and data flow dependencies
           for all 5 phases. This avoids repeated computation during evolution.

        2. **Bakes Operators**: Calls ``set_input_length()`` on each operator to freeze
           their static input sizes (population size, number of pairs, etc.). XLA uses
           this to generate optimized kernels.

        3. **Enforces Sharding**: Applies GSPMD layout to population genes, fitness,
           and individual tracking based on ``num_devices`` and ``pop_size``.

        4. **Initializes Population**: Calls ``genome_config.init_population()`` to
           create and evaluate the initial population. Returns the best individual.

        5. **Returns Cached State**: Packages the resource_map and baked operators into
           :class:`GeneticEvolutionState`. All :meth:`step` calls reuse these cached items.

        Parameters
        ----------
        rng_key : int or jax.Array
            Random seed. If an integer, a typed PRNG key is created using the engine's
            configured ``prng_impl`` (THREEFRY or PHILOX). If a jax.Array, it must be
            a valid JAX PRNG key (shape (2,) or newer typed format).

        Returns
        -------
        GeneticEvolutionState
            Initial state with fields:

            - ``population`` : Evaluated population (shape: (pop_size, ...genome_shape))
            - ``best_genome`` : Best individual found so far
            - ``best_fitness`` : Fitness of best_genome (scalar)
            - ``generation`` : 0
            - ``rng_key`` : PRNG key for next generation
            - ``resource_map`` : Cached plan for all phases (non-pytree)
            - ``operators`` : Baked selection/crossover/mutation ops (non-pytree)

        Notes
        -----
        **Consumes PRNG keys**: This method splits the input key to generate:
        - One key for population initialization
        - One key for the first generation in :meth:`step`

        **One-time cost**: ResourceMap computation and operator baking are done once;
        all :meth:`step` calls reuse the cached state.operators and state.resource_map.

        **JIT Compilation**: XLA compiles during this call, baking operator input sizes
        and sharding layout. Subsequent :meth:`step` calls do not recompile.

        Examples
        --------
        Initialize from integer seed::

            state = engine.init_state(rng_key=42)

        Initialize from pre-existing key::

            rng_key = jax.random.key(42)
            state = engine.init_state(rng_key=rng_key)
        """
        params = cast(GeneticEngineParams, self.engine_params)

        if params.debug_tracing:
            enable_tracing()
        else:
            disable_tracing()

        if isinstance(rng_key, int):
            rng_key = create_key(rng_key, impl=params.prng_impl)
        else:
            validate_key(rng_key, context="GeneticEngine.init_state()")

        typed = is_new_style_key(rng_key)

        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.engine_params.pop_size,
            params.elitism,
            params.key_derivation,
        )

        sharding_mgr = ShardingManager(axis_name="batch")
        active_sel = (
            cast(Any, self.selection)
            .replace(num_selections=rmap.selection.output_count)
            .set_input_length(rmap.selection.input_count)
            .set_typed_keys(typed)
            .set_n_elites(params.elitism)
        )

        active_cross = (
            self.crossover.set_input_length(rmap.crossover.input_count // 2)
            .set_typed_keys(typed)
            .set_max_generations(params.num_generations)
        )
        active_mut = (
            self.mutation.set_input_length(rmap.mutation.input_count)
            .set_typed_keys(typed)
            .set_max_generations(params.num_generations)
        )

        op_state = OperatorState(selection=active_sel, crossover=active_cross, mutation=active_mut)
        init_pop_key, rng_key = jar.split(rng_key)

        if not hasattr(self.genome_config, "init_population"):
            raise ValueError(
                f"Unsupported genome config: {type(self.genome_config).__name__}. "
                "Config must implement init_population(key, size) -> BasePopulation."
            )
        population = self.genome_config.init_population(init_pop_key, self.engine_params.pop_size)

        target_dtype = self.genome_config.dtype
        num_devices = len(sharding_mgr.devices)
        _pop_shardable = self.engine_params.pop_size % num_devices == 0

        def _enforce_layout(leaf: chex.Array) -> chex.Array:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                leaf = leaf.astype(target_dtype)

            if (
                _pop_shardable
                and hasattr(leaf, "shape")
                and len(leaf.shape) >= 2
                and leaf.shape[0] == self.engine_params.pop_size
            ):
                return jax.device_put(leaf, sharding_mgr.matrix_sharding)
            elif (
                _pop_shardable
                and hasattr(leaf, "shape")
                and len(leaf.shape) == 1
                and leaf.shape[0] == self.engine_params.pop_size
            ):
                return jax.device_put(leaf, sharding_mgr.vector_sharding)

            return jax.device_put(leaf, sharding_mgr.replicated_sharding)

        sharded_genes = jax.tree_util.tree_map(_enforce_layout, population.genes)
        fitness_casted = population.fitness.astype(target_dtype)
        sharded_fitness = jax.device_put(
            fitness_casted,
            sharding_mgr.vector_sharding if _pop_shardable else sharding_mgr.replicated_sharding,
        )

        population = population.replace(genes=sharded_genes, fitness=sharded_fitness)

        evaluated_pop = self.evaluator.evaluate_population(population)
        best_idx = jnp.argmin(evaluated_pop.fitness)

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
            resource_map=rmap,
            operators=op_state,
        )

    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """Generate next population batch for external evaluation (Ask/Tell interface).

        This method allocates entropy for the current generation and returns the
        population to be evaluated. Use :meth:`tell` to submit evaluated results
        and advance to the next generation.

        Parameters
        ----------
        state : GeneticEvolutionState
            Current evolution state (typically from a previous :meth:`tell` call
            or :meth:`init_state`).

        Returns
        -------
        engine_with_entropy : GeneticEngine
            Modified engine instance carrying allocated entropy in ``_entropy_buffer``.
            **IMPORTANT**: Use this returned engine (not self) in the subsequent
            :meth:`tell` call. The entropy is tied to this specific engine instance.

        population : BasePopulation
            Population ready for evaluation. Apply your fitness function to this.

        Notes
        -----
        **Ask/Tell Contract**: The ask/tell loop must follow this exact pattern:

        ::

            engine, population = engine.ask(state)
            # ... evaluate population externally ...
            evaluated_pop = evaluator.evaluate_population(population)
            new_state = engine.tell(state, evaluated_pop)

        **Critical**: The engine instance returned from :meth:`ask` carries entropy
        state in ``_entropy_buffer``. Only this engine can call :meth:`tell` for
        this generation. Do not use the original engine for :meth:`tell`.

        Examples
        --------
        Single-batch evaluation::

            engine, pop = engine.ask(state)
            fitness_scores = my_evaluator(pop)  # Custom evaluation
            evaluated_pop = pop.replace(fitness=fitness_scores)
            new_state = engine.tell(state, evaluated_pop)

        Parallel batch evaluation::

            batches = 4
            engines = [None] * batches
            populations = [None] * batches
            for i in range(batches):
                engines[i], populations[i] = engine.ask(state)

            # Distribute populations[i] to different GPUs/processes
            results = parallel_map(evaluate_fn, populations)

            # Aggregate results
            aggregated = populations[0].replace(fitness=results_aggregated)
            new_state = engines[0].tell(state, aggregated)
        """
        entropy = self._allocate_entropy(state)

        engine_with_entropy = cast(GeneticEngine, cast(Any, self).replace(_entropy_buffer=entropy))
        return engine_with_entropy, state.population

    def ask_with_key(
        self, state: GeneticEvolutionState, rng_key: chex.Array
    ) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """Ask variant with explicit key override for parity adapters.

        This mirrors ask/tell APIs that pass RNG at call time by temporarily
        overriding ``state.rng_key`` for entropy allocation in this generation.
        """
        state_with_key = cast(GeneticEvolutionState, cast(Any, state).replace(rng_key=rng_key))
        return self.ask(state_with_key)

    def tell(
        self, state: GeneticEvolutionState, population: BasePopulation[Any]
    ) -> GeneticEvolutionState:
        """Complete one generation using evaluated population (Ask/Tell interface).

        After calling :meth:`ask` and evaluating the returned population externally,
        pass the evaluated population here to complete the evolutionary step. This
        method executes selection, reproduction, merging, and HOF tracking.

        Parameters
        ----------
        state : GeneticEvolutionState
            Current evolution state (same one passed to prior :meth:`ask` call).
            The state is updated with the new population and generation counter.

        population : BasePopulation
            Evaluated population from :meth:`ask`. Must have ``fitness`` filled in
            by your external evaluator. Genomes should NOT be modified.
            Shape: (pop_size, ...genome_shape).

        Returns
        -------
        GeneticEvolutionState
            State after one generation of evolution. Contains:

            - ``population`` : New population after selection/reproduction/merge
            - ``best_genome`` : Updated best individual
            - ``best_fitness`` : Updated best fitness
            - ``generation`` : Incremented by 1
            - Other fields unchanged

        Raises
        ------
        RuntimeError
            If called before :meth:`ask` (entropy buffer is empty).

        Notes
        -----
        **Ask/Tell Contract**: See :meth:`ask` for the full loop pattern.

        **Engine Instance**: Use the engine instance returned from :meth:`ask`,
        not the original self.

        **No Compilation**: XLA compilation happened in :meth:`init_state`. This call
        reuses cached compiled kernels.

        Examples
        --------
        Complete ask/tell loop::

            # Initialize once
            state = engine.init_state(rng_key=42)

            # Repeat for each generation
            for gen in range(num_generations):
                engine, pop = engine.ask(state)
                # Evaluate in parallel, custom evaluator, etc.
                scores = external_fitness_fn(pop)
                evaluated = pop.replace(fitness=scores)
                state = engine.tell(state, evaluated)

            # Access final results
            print(f"Best: {state.best_fitness}")
            print(f"Generations run: {state.generation}")
        """
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask().")

        k_sel, k_cross, k_mut, k_next = self._entropy_buffer

        state = cast(GeneticEvolutionState, cast(Any, state).replace(population=population))
        gen_best_fitness = jnp.min(population.fitness)
        is_new = gen_best_fitness < state.best_fitness
        new_best_fitness = jnp.where(is_new, gen_best_fitness, state.best_fitness)
        best_idx = jnp.argmin(population.fitness)
        best_candidate = jax.tree_util.tree_map(lambda x: x[best_idx], population.genes)
        new_best_genome = jax.tree_util.tree_map(
            lambda n, o: jnp.where(is_new, n, o),
            best_candidate,
            state.best_genome,
        )

        state = cast(
            GeneticEvolutionState,
            cast(Any, state).replace(
                best_genome=new_best_genome,
                best_fitness=new_best_fitness,
            ),
        )

        elites, parent_indices = self._selection_phase(
            k_sel, state.population, state.operators, self.engine_params
        )

        mutants = self._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            state.population,
            state.operators,
            state.resource_map,
            generation=state.generation,
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

        # Clear the engine-side entropy buffer to prevent accidental reuse of
        # the same entropy for subsequent `tell()` calls. We attempt to clear
        # the frozen flax.struct instance via object.__setattr__; if the
        # underlying object forbids mutation this is non-fatal and we simply
        # leave the buffer as-is (the ask/tell contract still enforces usage
        # discipline via runtime checks above).
        try:
            object.__setattr__(self, "_entropy_buffer", ())
        except Exception:
            # Best-effort: if we cannot mutate, don't crash the evolution.
            pass

        return final_state

    def tell_with_key(
        self, state: GeneticEvolutionState, population: BasePopulation[Any], rng_key: chex.Array
    ) -> GeneticEvolutionState:
        """Tell variant with explicit key override for parity adapters.

        The current ask/tell implementation pre-allocates all entropy in
        :meth:`ask`; therefore ``rng_key`` is accepted for interface parity and
        intentionally not consumed here.
        """
        _ = rng_key
        return self.tell(state, population)
