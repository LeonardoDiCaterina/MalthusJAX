from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""

G = TypeVar("G", bound=BaseGenome)  # Genome type
P = TypeVar("P", bound=BasePopulation[Any])  # Population type (parameterized with Any)

_field: Any = struct.field  # Helper alias for typed field calls


def compute_unroll_num(num_generations: int) -> int:
    """Return the scan unroll factor for an evolution loop.

    Historically this adjusted XLA scan unrolling, but the mechanism was
    deprecated after benchmarks showed no benefit and a linear growth in IR
    size. It remains here for backwards compatibility and now always returns
    ``1`` regardless of *num_generations*.
    """
    return 1


def validate_engine_params(params: "AbstractEngineParams") -> None:
    """
    Validate engine parameters outside of JIT context.
    Call this before starting evolution to catch configuration errors early.
    """
    if params.pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {params.pop_size}")

    if params.num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {params.num_generations}")

    # Elitism checks (optional based on algorithm)
    if not (0 <= params.elitism < params.pop_size):
        raise ValueError(
            f"elitism must satisfy 0 <= elitism < pop_size, "
            f"got elitism={params.elitism}, pop_size={params.pop_size}"
        )


@struct.dataclass
class AbstractEngineParams:
    """
    Base immutable configuration for evolution engines (pytree_node=False).
    All fields remain static during JIT compilation; mutations use .replace().
    - pop_size: Population size (must be > 0).
    - elitism: Number of elite individuals preserved each generation (0 ≤ elitism < pop_size).
    - num_generations: Number of evolution steps (must be > 0).
    - unroll_num: JAX scan unroll factor for latency/memory trade-off (default 1).
    """

    pop_size: int = _field(pytree_node=False, default=100)
    elitism: int = _field(pytree_node=False, default=0)
    num_generations: int = _field(pytree_node=False, default=50)
    unroll_num: int = _field(pytree_node=False, default=1)


@struct.dataclass
class AbstractEvolutionState(Generic[G, P]):
    """
    Mutable state container for evolution across generations (carries data through scan).
    Concrete implementations (GeneticEvolutionState) extend this with resource_map, operators.
    
    Type System Note:
    As of v2.0, the population generic parameter `P` has been decoupled from the
    operators but is preserved here for static typing of the scan state. It
    universally binds to `BasePopulation[G]` or a structural subclass like
    `RealPopulation`. The engine is completely agnostic of the genome's internal
    structure (e.g., `.values`).

    - population (P): Current population (shape: (pop_size, ...genome_shape)).
    - best_genome (G): Best individual found so far (shape: (...genome_shape)).
    - generation (int): Current generation counter (increments each step).
    - best_fitness (Array): Scalar fitness of best_genome.
    - rng_key (Array): Master PRNG key for next generation (shape: (2,)).
    """

    population: P
    best_genome: G

    generation: int
    best_fitness: chex.Array
    rng_key: chex.Array


@struct.dataclass
class AbstractGenerationOutput:
    """
    KPI payload returned at every evolution step (collected by JAX scan).
    Foundation for universal dashboard generation.
    - best_fitness: Scalar best fitness this generation.
    - mean_fitness: Scalar population mean fitness this generation.
    - generation: Current generation counter.
    History: scan returns tuple of (final_state, jax.lax.scan output), where output
    stacks all KPI instances into (num_generations,) shaped arrays per field.
    """

    best_fitness: chex.Array
    mean_fitness: chex.Array
    std_fitness: chex.Array
    generation: chex.Array

    @classmethod
    def get_kpi_names(cls) -> List[str]:
        return list(getattr(cls, "__dataclass_fields__", {}).keys())


@struct.dataclass
class AbstractEngine(Generic[G, P], ABC):
    """
    Abstract base class for all evolutionary engines.

    Standardizes the evolution loop using JAX scan.
    """

    engine_params: AbstractEngineParams = _field(pytree_node=False)

    def __hash__(self) -> int:
        """Make engine hashable for JIT static_argnums.

        Uses id() since engine contains JAX arrays that aren't hashable.
        This means each engine instance is treated as unique by JIT caching.
        """
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other

    @abstractmethod
    def init_state(self, rng_key: Union[int, jnp.ndarray]) -> AbstractEvolutionState[G, P]:
        """Build and return the initial evolution state.

        This first-phase compilation step sets up the resource mapper, operators,
        and initial population. If *rng_key* is an integer, implementations
        should convert it to a typed PRNG key using the configured backend.
        The returned state is cached and reused by ``step`` and ``run``.
        """
        validate_engine_params(self.engine_params)
        raise NotImplementedError

    @abstractmethod
    def step(
        self, state: AbstractEvolutionState[G, P]
    ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput]:
        """Perform a single evolutionary generation update.

        Engines typically perform entropy allocation, selection, reproduction,
        merging, evaluation, and high‑order‑function updates. The input *state*
        carries the current population, PRNG key, and other resources. The
        method returns the updated state together with a KPI payload.
        """
        raise NotImplementedError

    def debug_step(
        self, state: AbstractEvolutionState[G, P]
    ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput]:
        """Run one step and print the resulting population length."""
        new_state, output = self.step(state)
        population = getattr(new_state, "population", None)
        if population is not None:
            print(f"population len: {len(population)}")
        return new_state, output

    def debug_run(
        self, initial_state: AbstractEvolutionState[G, P]
    ) -> Tuple[AbstractEvolutionState[G, P], List[AbstractGenerationOutput]]:
        """Run the engine in a Python loop using `debug_step`."""
        state = initial_state
        history: List[AbstractGenerationOutput] = []
        for _ in range(self.engine_params.num_generations):
            state, output = self.debug_step(state)
            history.append(output)
        return state, history

    def run(
        self,
        initial_state: AbstractEvolutionState[G, P],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
    ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput, Optional[float]]:
        """Run the full evolution loop starting from *initial_state*.

        This method obtains the JIT‑compiled scan kernel and executes it for
        the configured number of generations. If *time_it* is true the method
        measures wall‑clock duration, optionally printing progress when
        *verbose* is enabled. Returns the final state, the stacked KPI history,
        and an optional elapsed time.
        """
        if verbose:
            print(
                f"Starting evolution: {self.engine_params.num_generations} generations, "
                f"population size {self.engine_params.pop_size}, compile={compile}"
            )
        # Retrieve the compiled loop function
        evolve_fn = _get_evolution_kernel(
            self.engine_params, compile_jit=compile, unroll_num=self.engine_params.unroll_num
        )

        start_time = time.time()

        try:
            # Execute Scan
            final_state, history = evolve_fn(self, initial_state)

            # Force synchronization for accurate timing if requested
            if time_it:
                _ = final_state.best_fitness.block_until_ready()

        except Exception as e:
            raise RuntimeError(f"Evolution failed: {str(e)}") from e

        elapsed_time = None
        if time_it:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"Done in {elapsed_time:.3f}s")

        return final_state, history, elapsed_time

    def ask_with_key(
        self, state: AbstractEvolutionState[G, P], rng_key: chex.Array
    ) -> Tuple["AbstractEngine[G, P]", P]:
        """Optional key-aware ask interface.

        Default behavior delegates to ``ask(state)`` when implemented by the
        concrete engine. Engines that need explicit key control can override
        this method.
        """
        _ = rng_key
        ask_fn = getattr(self, "ask", None)
        if callable(ask_fn):
            return cast(Tuple["AbstractEngine[G, P]", P], ask_fn(state))
        raise NotImplementedError(
            "ask_with_key() is not implemented for this engine and no ask() method is available."
        )

    def tell_with_key(
        self, state: AbstractEvolutionState[G, P], population: P, rng_key: chex.Array
    ) -> AbstractEvolutionState[G, P]:
        """Optional key-aware tell interface.

        Default behavior delegates to ``tell(state, population)`` when
        implemented by the concrete engine. Engines that need explicit key
        control can override this method.
        """
        _ = rng_key
        tell_fn = getattr(self, "tell", None)
        if callable(tell_fn):
            return cast(AbstractEvolutionState[G, P], tell_fn(state, population))
        raise NotImplementedError(
            "tell_with_key() is not implemented for this engine and no tell() method is available."
        )

    def get_hlo_text(
        self,
        initial_state: AbstractEvolutionState[G, P],
        optimize: bool = True,  # <--- NEW FLAG
        print_analysis: bool = True,
    ) -> str:
        """
        Extracts HLO text.
        If optimize=True, runs the full XLA compiler to show FUSION.
        If optimize=False, shows the raw graph (faster, good for debugging shapes).
        """
        print(f"--- Extracting HLO (Optimize={optimize}) ---")

        # 1. Get JIT Kernel
        jit_kernel = _get_evolution_kernel(
            self.engine_params, compile_jit=True, unroll_num=self.engine_params.unroll_num
        )

        # 2. Lower (Trace to StableHLO)
        lowered = jit_kernel.lower(self, initial_state)

        # 3. Compile (Run XLA Optimizations)
        if optimize:
            # This triggers the fusion strategies!
            compiled = lowered.compile()
            hlo_text = cast(str, compiled.as_text())
        else:
            hlo_text = cast(str, lowered.as_text())

        if print_analysis:
            line_count = len(hlo_text.split("\n"))
            fusion_count = hlo_text.count("fusion")
            loop_count = hlo_text.count("while")
            # Note: 'while' may be transformed (e.g., to 'custom-call') by the backend optimizations

            print("HLO Analysis:")
            print(f"  - Total Lines of IR: {line_count}")
            print(f"  - Fusion Kernels:    {fusion_count} (Higher is better)")
            print(f"  - Explicit Loops:    {loop_count} (Should be 1 for the main scan)")
            print("-" * 30)

        return hlo_text


def _get_evolution_kernel(
    params: AbstractEngineParams, compile_jit: bool = True, unroll_num: int = 1
) -> Any:
    """
    Factory: Builds evolution kernel (jax.lax.scan loop).
    JAX's own compilation cache (keyed on static args + input shapes) handles
    deduplication — an additional lru_cache is redundant and can leak memory
    by preventing GC of closed-over engine/state references (JR-3).
    Closure pattern: _evolve_loop captures engine as compile-time constant (static_argnums=0).
    This avoids passing engine in scan carry ("light carry"), reducing memory.
    donate_argnums=1 donates initial_state arrays (JIT donation optimization).
    unroll_num: Deprecated no-op — always overridden to 1.  Benchmarks show
    scan unrolling grows XLA IR linearly (unroll_5 → 2.3×, unroll_25 → 9×)
    with zero throughput benefit on GPU.  XLA fuses across scan iterations
    automatically.
    """
    if unroll_num != 1:
        import warnings

        warnings.warn(
            f"unroll_num={unroll_num} has no performance benefit on GPU and increases "
            "XLA compile time linearly. It has been overridden to 1. "
            "See GeneticEngineParams docstring for details.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _evolve_loop(
        engine: AbstractEngine[G, P], initial_state: AbstractEvolutionState[G, P]
    ) -> Tuple[AbstractEvolutionState[G, P], Any]:
        def _scan_body_closure(
            state: AbstractEvolutionState[G, P], __: Any
        ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput]:
            new_state, history_item = engine.step(state)
            return new_state, history_item

        init_carry = initial_state

        final_state, history = jax.lax.scan(
            _scan_body_closure,
            init_carry,
            None,
            length=params.num_generations,
            unroll=unroll_num,
        )

        return final_state, history

    if compile_jit:
        return jax.jit(_evolve_loop, donate_argnums=1, static_argnums=0)
    else:
        return _evolve_loop
