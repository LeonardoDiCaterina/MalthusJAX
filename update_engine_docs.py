#!/usr/bin/env python3
"""Update engine.py docstrings comprehensively."""


file_path = 'src/malthusjax/engine/genetic_fastengine.py'

with open(file_path, 'r') as f:
    content = f.read()

# ===============================================================================
# 1. Update GeneticEngineParams docstring
# ===============================================================================
params_old = '''@struct.dataclass
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
    """'''

params_new = '''@struct.dataclass
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
    """'''

content = content.replace(params_old, params_new)
print("✅ Updated GeneticEngineParams")

# ===============================================================================
# 2. Update GeneticEngine class docstring
# ===============================================================================
engine_old = '''@struct.dataclass
class GeneticEngine(AbstractEngine[BaseGenome, BasePopulation[Any]]):
    """
    High-Performance Genetic Algorithm Engine (Init-Phase Compilation).
    Architecture: init_state() compiles ResourceMap, bakes operators with static input sizes,
    returns state carrying pre-baked tools. step() executes using cached resource plan.
    5-Phase per generation: (0) Entropy allocation → (1) Selection → (2) Reproduction →
    (3) Merge/Elite preservation → (4) Evaluate + HOF update.
    Ask/Tell interface: Alternative injection-style control flow (ask entropy, tell evaluated pop).
    """'''

engine_new = '''@struct.dataclass
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

    See :class:`GeneticEngineParams` for all tunable configuration parameters.
    """'''

content = content.replace(engine_old, engine_new)
print("✅ Updated GeneticEngine class docstring")

# ===============================================================================
# 3. Enhance init_state docstring
# ===============================================================================
init_old = '''    def init_state(self, rng_key: Union[int, chex.Array]) -> GeneticEvolutionState:
        """
        Initialize evolution state (Init-Phase Compilation).

        Accepts either an integer seed (convenience) or a pre-constructed PRNG key.
        If an integer seed is provided, a typed key is created using the engine's
        configured `prng_impl`. If a legacy `PRNGKey` is provided, a
        ``DeprecationWarning`` is emitted.

        Steps:
        (1) Compute ResourceMap (RNG budget + data flow cascade).
        (2) Bake operators: set_input_length freezes static sizes for XLA.
        (3) Enforce GSPMD sharding layout (per-device or replicated).
        (4) Initialize and evaluate population.
        (5) Return state carrying resource_map + operators for all steps().
        One-time cost; results cached in state.resource_map throughout run.
        """'''

init_new = '''    def init_state(self, rng_key: Union[int, chex.Array]) -> GeneticEvolutionState:
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
        """'''

content = content.replace(init_old, init_new)
print("✅ Updated init_state docstring")

# ===============================================================================
# 4. Enhance run docstring
# ===============================================================================
run_old = '''    def run(
        self,
        initial_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
    ) -> Tuple[
        AbstractEvolutionState[BaseGenome, BasePopulation[Any]], AbstractGenerationOutput, Any
    ]:
        """Execute evolution and apply post-scan finalization.

        Delegates the main scan loop to ``AbstractEngine.run()``, then
        populates ``best_genome`` (and ``best_fitness`` for NONE mode)
        from the final population.  This one-shot O(N) ``argmax``
        replaces the per-step Gather that LIGHT/NONE modes skip.
        """'''

run_new = '''    def run(
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

        **Hall-of-Fame Tracking**: Best individual tracking strategy (from engine_params.track_best)—

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
        """'''

content = content.replace(run_old, run_new)
print("✅ Updated run docstring")

# ===============================================================================
# 5. Enhance ask/tell docstrings
# ===============================================================================
ask_old = '''    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """
        Ask for next evaluation batch (injection-style interface).
        Allocates entropy for this generation, returns (engine_with_entropy, population).
        Call pattern: engine, pop = ask(state); evaluated_pop = evaluate(pop);
        new_state = tell(state, evaluated_pop).
        Alternative to step() for external fitness evaluators.
        """'''

ask_new = '''    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation[Any]]:
        """Generate next population batch for external evaluation (Ask/Tell interface).

        This method allocates entropy for the current generation and returns the
        population to be evaluated. Use :meth:`tell` to submit evaluated results
        and advance to the next generation.

        Parameters
        ----------
        state : GeneticEvolutionState
            Current evolution state (typically from previous :meth:`tell` call or :meth:`init_state`).

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
        """'''

content = content.replace(ask_old, ask_new)
print("✅ Updated ask docstring")


tell_old = '''    def tell(
        self, state: GeneticEvolutionState, population: BasePopulation[Any]
    ) -> GeneticEvolutionState:
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask().")'''

tell_new = '''    def tell(
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
            raise RuntimeError("tell() called before ask().")'''

content = content.replace(tell_old, tell_new)
print("✅ Updated tell docstring")

# Write the updated file
with open(file_path, 'w') as f:
    f.write(content)

print("\n" + "="*70)
print("✅ All engine.py docstrings updated successfully!")
print("="*70)
