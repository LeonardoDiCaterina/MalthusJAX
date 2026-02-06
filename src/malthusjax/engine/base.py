"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""
import functools
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, cast

import chex
import flax.struct
import jax
import jax.numpy as jnp
from flax import struct

# Import for generic typing - ensure these exist in your project
from malthusjax.core.base import BaseGenome, BasePopulation

# Type variables for generics
G = TypeVar('G', bound=BaseGenome)        # Genome type
P = TypeVar('P', bound=BasePopulation[Any])    # Population type

_field: Callable[..., Any] = struct.field  # Alias for flax.struct.field
def validate_engine_params(params: 'AbstractEngineParams') -> None:
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


@flax.struct.dataclass
class AbstractEngineParams:
    """
    Base immutable configuration for evolution engines.
    All fields are marked as pytree_node=False to ensure they remain
    static during JIT compilation.
    """
    pop_size: int = _field(pytree_node=False, default=100)
    elitism: int = _field(pytree_node=False, default=0)
    num_generations: int = _field(pytree_node=False, default=50)
    unroll_num: int = _field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        # make unroll_num a 10% of num_generations if possible
        object.__setattr__(self, 'unroll_num',
                           min(max(1, self.num_generations // 10), self.num_generations))



@flax.struct.dataclass
class AbstractEvolutionState(Generic[P, G]):
    """
    Mutable state container that evolves across generations.
    
    Concrete implementations (like GeneticEvolutionState) will extend this
    to add 'resource_map' and 'operators'.
    """
    # --- CRITICAL: Population and Best Individual ---
    population: P
    best_genome: G

    # --- Metadata ---
    generation: int
    best_fitness: chex.Array
    stagnation_counter: int
    rng_key: chex.Array


@flax.struct.dataclass
class AbstractGenerationOutput:
    """
    Base KPI payload returned at every evolution step.
    Foundation for universal dashboard generation.
    """
    best_fitness: chex.Array
    mean_fitness: chex.Array
    generation: chex.Array

    @classmethod
    def get_kpi_names(cls) -> List[str]:
        return list(cls.__dataclass_fields__.keys())


@struct.dataclass
class AbstractEngine(Generic[P, G], ABC):
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

    def __eq__(self,
               other: object) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other

    @abstractmethod
    def init_state(self,
                   rng_key: jnp.ndarray
                   ) -> AbstractEvolutionState[P, G]:
        """
        Initialize the evolution state (Compile Plan & Bake Operators).
        Note: We removed 'params' from arg list because self.engine_params exists.
        """
        # Input validation should happen here or in constructor
        validate_engine_params(self.engine_params)
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: AbstractEvolutionState[P, G]
    ) -> Tuple[AbstractEvolutionState[P, G], AbstractGenerationOutput]:
        """
        Execute one generation step.
        Args:
            state: Current state (containing rng_key, operators, population).
        Returns:
            (new_state, history_item)
        """
        raise NotImplementedError

    def run(
        self,
        initial_state: AbstractEvolutionState[P, G],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False
    ) -> Tuple[AbstractEvolutionState[P, G], AbstractGenerationOutput, Optional[float]]:
        """
        Run complete evolution using JAX scan pattern.
        """
        if verbose:
            print(f"Starting evolution: {self.engine_params.num_generations} generations, "
                  f"population size {self.engine_params.pop_size}, compile={compile}")

        evolve_fn = _get_evolution_kernel(self.engine_params, compile_jit=compile, unroll_num=self.engine_params.unroll_num)
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
        initial_state: AbstractEvolutionState[P, G],
        optimize: bool = True,
        print_analysis: bool = True
    ) -> str:
        """
        Extracts HLO text. 
        If optimize=True, runs the full XLA compiler to show FUSION.
        If optimize=False, shows the raw graph (faster, good for debugging shapes).
        """
        print(f"--- Extracting HLO (Optimize={optimize}) ---")
        jit_kernel = _get_evolution_kernel(self.engine_params, compile_jit=True, unroll_num=self.engine_params.unroll_num)
        lowered = jit_kernel.lower(self, initial_state)
        if optimize:
            # This triggers the fusion strategies!
            compiled = lowered.compile()
            hlo_text = compiled.as_text()
        else:
            hlo_text = lowered.as_text()
        if print_analysis:
            line_count = len(hlo_text.split('\n'))
            fusion_count = hlo_text.count("fusion")
            loop_count = hlo_text.count("while")
            # Note: In optimized HLO, 'while' might become 'custom-call' or similar depending on backend
            print("HLO Analysis:")
            print(f"  - Total Lines of IR: {line_count}")
            print(f"  - Fusion Kernels:    {fusion_count} (Higher is better)")
            print(f"  - Explicit Loops:    {loop_count} (Should be 1 for the main scan)")
            print("-" * 30)
        return cast(str, hlo_text)

@functools.lru_cache(maxsize=32)
def _get_evolution_kernel(
    params: AbstractEngineParams,
    compile_jit: bool = True,
    unroll_num: int = 1
    ) -> Any:
    """
    Factory that builds and compiles the evolution loop.
    Cached by 'params' to ensure we only compile once per configuration.
    """
    def _evolve_loop(engine: AbstractEngine[P, G], initial_state: AbstractEvolutionState[P, G]) -> Tuple[AbstractEvolutionState[P, G], AbstractGenerationOutput]:
        """
        Core evolution loop using JAX scan.
        """
        # Because we are inside _evolve_loop, 'engine' is available in the scope.
        # We do NOT need to pass it in the carry.
        def _scan_body_closure(state: AbstractEvolutionState[P, G],
                                _: None
                               ) -> Tuple[AbstractEvolutionState[P, G], AbstractGenerationOutput]:
            """
            Scan body that calls engine.step().
            Uses closure to access 'engine'.
            """
            # 'engine' is a compile-time constant here because
            # we use static_argnums=0 on the outer function.
            new_state, history_item = engine.step(state)
            # Return ONLY state, no engine in the tuple!
            return new_state, history_item
        # Carry is just the state. The backpack is light!
        init_carry = initial_state

        final_state, history = jax.lax.scan(
            _scan_body_closure,      # <--- Uses the closure
            init_carry,
            None,
            length=params.num_generations,
            unroll=unroll_num
        )
        return final_state, history
    # 3. JIT Compile
    if compile_jit:
        # static_argnums=0 is CRITICAL.
        # It tells JAX: "engine is not data, it is the program logic."
        return jax.jit(_evolve_loop, donate_argnums=1, static_argnums=0)
    else:
        return _evolve_loop




'''
src/malthusjax/engine/base.py:22: error: Type variable "malthusjax.engine.base.G" is unbound  [valid-type]
src/malthusjax/engine/base.py:22: note: (Hint: Use "Generic[G]" or "Protocol[G]" base class to bind "G" inside a class)
src/malthusjax/engine/base.py:22: note: (Hint: Use "G" in function signature to bind "G" inside a function)
src/malthusjax/engine/base.py:52: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
src/malthusjax/engine/base.py:53: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
src/malthusjax/engine/base.py:54: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
src/malthusjax/engine/base.py:55: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
src/malthusjax/engine/base.py:104: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
src/malthusjax/engine/base.py:113: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
src/malthusjax/engine/base.py:118: error: Missing return statement  [return]
src/malthusjax/engine/base.py:118: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:129: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:130: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:142: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:146: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:173: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:201: error: Returning Any from function declared to return "str"  [no-any-return]
src/malthusjax/engine/base.py:203: error: Function is missing a return type annotation  [no-untyped-def]
src/malthusjax/engine/base.py:208: error: Function is missing a return type annotation  [no-untyped-def]
src/malthusjax/engine/base.py:208: error: Missing type parameters for generic type "AbstractEvolutionState"  [type-arg]
src/malthusjax/engine/base.py:212: error: Function is missing a type annotation  [no-untyped-def]
'''
