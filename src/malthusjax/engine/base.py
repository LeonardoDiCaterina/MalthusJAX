from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar, cast, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Import for generic typing - ensure these exist in your project
from malthusjax.core.base import BaseGenome, BasePopulation

"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""

# Type variables for generics
G = TypeVar("G", bound=BaseGenome)  # Genome type
P = TypeVar("P", bound=BasePopulation[Any])  # Population type (parameterized with Any)

_field: Any = struct.field  # Helper alias for typed field calls


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
    - unroll_num: JAX scan unroll factor for latency/memory trade-off.
      Auto-tuned to 10% of num_generations (clamped [1, num_generations]).
      Smaller = more memory, higher latency; larger = less memory, more memory usage.
    """

    pop_size: int = _field(pytree_node=False, default=100)
    elitism: int = _field(pytree_node=False, default=0)
    num_generations: int = _field(pytree_node=False, default=50)
    unroll_num: int = _field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        # make unroll_num a 10% of num_generations if possible
        object.__setattr__(
            self, "unroll_num", min(max(1, self.num_generations // 10), self.num_generations)
        )


@struct.dataclass
class AbstractEvolutionState(Generic[G, P]):
    """
    Mutable state container for evolution across generations (carries data through scan).
    Concrete implementations (GeneticEvolutionState) extend this with resource_map, operators.
    - population (P): Current population (shape: (pop_size, ...genome_shape)).
    - best_genome (G): Best individual found so far (shape: (...genome_shape)).
    - generation (int): Current generation counter (increments each step).
    - best_fitness (Array): Scalar fitness of best_genome.
    - stagnation_counter (int): Generations since improvement (resets on new best).
    - rng_key (Array): Master PRNG key for next generation (shape: (2,)).
    """

    # --- CRITICAL: Population and Best Individual ---
    population: P
    best_genome: G

    # --- Metadata ---
    generation: int
    best_fitness: chex.Array
    stagnation_counter: int
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
        """
        Initialize evolution state (Init-Phase Compilation).

        Accepts either an integer seed or a pre-constructed PRNG key. If a seed is
        provided the concrete engine implementation is expected to create a typed
        PRNG key (e.g., using ``malthusjax.core.random.create_key``) based on its
        configuration.

        Responsibilities: Compile ResourceMap, bake operators with static input sizes,
        enforce sharding layout, initialize population, evaluate, identify best.
        One-time cost; results cached in returned state for all step() calls.
        Returns: Initial state ready for run() or step() iteration.
        """
        validate_engine_params(self.engine_params)
        raise NotImplementedError

    @abstractmethod
    def step(
        self, state: AbstractEvolutionState[G, P]
    ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput]:
        """
        Execute one generation (called per scan iteration).
        Generic engines implement: Entropy allocation → Selection → Reproduction →
        Merge → Evaluate → HOF update.
        Args: state (contains population, rng_key, resource_map, operators).
        Returns: (new_state with updated population/best/generation, KPI metrics).
        """
        raise NotImplementedError

    def run(
        self,
        initial_state: AbstractEvolutionState[G, P],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
    ) -> Tuple[AbstractEvolutionState[G, P], AbstractGenerationOutput, Optional[float]]:
        """
        Execute complete evolution via JAX scan (num_generations iterations).
        Retrieves cached/compiled evolution kernel, executes scan, returns final state + history.
        History: JAX scan output stacks all GenerationOutput KPIs across generations.
        Args: initial_state (from init_state), compile flag, timing/verbose options.
        Returns: (final_state, history_kpis, elapsed_time_or_none).
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


@functools.lru_cache(maxsize=32)
def _get_evolution_kernel(
    params: AbstractEngineParams, compile_jit: bool = True, unroll_num: int = 1
) -> Any:
    """
    Factory: Builds and caches evolution kernel (jax.lax.scan loop).
    Caching: lru_cache by params ensures one compilation per config.
    Closure pattern: _evolve_loop captures engine as compile-time constant (static_argnums=0).
    This avoids passing engine in scan carry ("light carry"), reducing memory.
    donate_argnums=1 donates initial_state arrays (JIT donation optimization).
    unroll_num: Scan unroll factor (latency vs memory trade-off).
    """

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
