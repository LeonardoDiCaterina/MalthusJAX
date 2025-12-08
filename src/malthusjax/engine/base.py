"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.

NOTE: To enable persistent disk caching (saving compilation time across restarts),
add the following to your main script before running:
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
"""

import jax
import chex
import jax.numpy as jnp 
import jax.random as jar 
import flax.struct 
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any, List, Generic, TypeVar
import functools
import time

# Import for generic typing
from malthusjax.core.base import BaseGenome, BasePopulation

# Type variables for generics
G = TypeVar('G', bound=BaseGenome)  # Genome type
P = TypeVar('P', bound=BasePopulation)  # Population type


def validate_engine_params(params: 'AbstractEngineParams') -> None:
    """
    Validate engine parameters outside of JIT context.
    Call this before starting evolution to catch configuration errors early.
    """
    if params.pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {params.pop_size}")
    
    if params.num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {params.num_generations}")
    
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
    static during JIT compilation. This is CRITICAL for JAX's persistent
    compilation cache to work correctly.
    
    Attributes:
        pop_size: Population size (must be positive)
        elitism: Number of elite individuals to preserve (0 <= elitism < pop_size)
        num_generations: Total generations to evolve (must be positive)
    """
    pop_size: int = flax.struct.field(pytree_node=False, default=100)
    elitism: int = flax.struct.field(pytree_node=False, default=0)
    num_generations: int = flax.struct.field(pytree_node=False, default=50)


@flax.struct.dataclass
class AbstractEvolutionState(Generic[G, P]):
    """
    Mutable state container that evolves across generations.
    Must contain only JAX-compatible types for JIT compilation.
    """
    # --- CRITICAL: Population and Best Individual ---
    population: P  # Current population
    best_genome: G  # Best genome found so far
    
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
        """Return available KPI field names for visualization."""
        return list(cls.__dataclass_fields__.keys())
    
    def get_kpi_value(self, kpi_name: str) -> chex.Array:
        """Extract specific KPI value by name."""
        if kpi_name not in self.get_kpi_names():
            raise AttributeError(
                f"KPI '{kpi_name}' not found. Available KPIs: {self.get_kpi_names()}"
            )
        return getattr(self, kpi_name)


class AbstractHook:
    """
    Strategy pattern interface for evolution callbacks.
    Enables clean extension points without breaking JIT compilation.
    """
    def __call__(self, state: AbstractEvolutionState, params: AbstractEngineParams) -> AbstractEvolutionState:
        """
        Must return modified (or same) state.
        Must be JIT-compatible (no side effects).
        """
        return state


class NoOpHook(AbstractHook):
    """Default no-operation hook"""
    def __call__(self, state: AbstractEvolutionState, params: AbstractEngineParams) -> AbstractEvolutionState:
        return state


class AbstractEngine(ABC):
    """
    Abstract base class for all evolutionary engines.
    
    Standardizes the evolution loop using JAX scan.
    Leverages JAX's internal compilation cache (memory & disk) automatically.
    """
    
    def __init__(self):
        """Initialize engine."""
        pass
    
    @abstractmethod
    def init_state(self, rng_key: jnp.ndarray, params: AbstractEngineParams) -> AbstractEvolutionState:
        """
        Initialize the evolution state.
        """
        pass
    
    @abstractmethod
    def step(
        self, 
        key: jnp.ndarray, 
        state: AbstractEvolutionState, 
        params: AbstractEngineParams
    ) -> Tuple[jnp.ndarray, AbstractEvolutionState, AbstractGenerationOutput]:
        """
        Execute one generation step.
        """
        pass
    
    def run(
        self, 
        initial_state: AbstractEvolutionState, 
        params: AbstractEngineParams,
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False
    ) -> Tuple[AbstractEvolutionState, AbstractGenerationOutput, Optional[float]]:
        """
        Run complete evolution using JAX scan pattern.
        
        This method orchestrates the full evolutionary loop. It relies on JAX's 
        built-in JIT compilation caching. 
        
        Performance Tip:
        To enable persistent disk caching (saving compilation time across process restarts),
        configure JAX before running this method:
            jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        
        Args:
            initial_state: Initial evolution state with population and RNG key
            params: Engine parameters (pop_size, num_generations, etc.)
            time_it: If True, measure and return execution time
            compile: If True, use JIT compilation (recommended)
            verbose: If True, print progress and timing information
            
        Returns:
            Tuple of (final_state, history, elapsed_time)
        """
        # Input validation
        validate_engine_params(params)
        
        if not isinstance(initial_state, AbstractEvolutionState):
            raise TypeError(
                f"initial_state must be AbstractEvolutionState, got {type(initial_state)}"
            )
        
        if not isinstance(params, AbstractEngineParams):
            raise TypeError(
                f"params must be AbstractEngineParams, got {type(params)}"
            )
        
        if verbose:
            print(f"Starting evolution: {params.num_generations} generations, "
                  f"population size {params.pop_size}, compile={compile}")
        
        start_time = time.time() if time_it else None
        
        # --- 1. Define the Scan Body ---
        # We define this inside run() to capture 'self' and 'params'.
        # Because 'params' fields are static (pytree_node=False), JAX treats
        # this closure as a stable function signature suitable for caching.
        def _scan_body(state, _):
            
            rng_key = state.rng_key
            
            # The engine's step() handles key splitting internally.
            # We pass the full 'params' struct, which acts as static config.
            _, new_state, history_item = self.step(rng_key, state, params)            
            return new_state, history_item

        # --- 2. Define the Execution Wrapper ---
        def _evolve_loop(init_carry):
            return jax.lax.scan(
                _scan_body,
                init_carry,
                None,
                length=params.num_generations
            )

        # --- 3. JIT Compilation ---
        # If compile=True, we wrap the loop in jit.
        if compile:
            # Use functools.lru_cache to cache the JIT-compiled function
            # This prevents recompilation when run() is called multiple times
            # with the same engine instance and parameter structure
            @functools.lru_cache(maxsize=1)
            def _get_jitted_fn():
                return jax.jit(_evolve_loop, donate_argnums=0)

            evolution_fn = _get_jitted_fn()
        else:
            evolution_fn = _evolve_loop

        # --- 4. Execution ---
            
        try:
            # First execution triggers JIT compilation (trace) if not cached.
            final_state, history = evolution_fn(initial_state)
            
            # If timing, we must block until computation finishes on device
            if time_it:
                jax.block_until_ready(final_state)
                
        except Exception as e:
            raise RuntimeError(
                f"Evolution loop failed at generation {initial_state.generation}: {str(e)}"
            ) from e    
        
        # --- 5. Finalization ---
        elapsed_time = None
        if time_it:
            elapsed_time = time.time() - start_time
            if verbose:
                gen_time = elapsed_time / params.num_generations
                print(f"Evolution completed in {elapsed_time:.3f}s "
                      f"({gen_time*1000:.2f}ms/gen)")
        
        if verbose:
            print(f"Final generation: {final_state.generation}")
            print(f"Best fitness: {float(final_state.best_fitness):.6f}")
        
        return final_state, history, elapsed_time