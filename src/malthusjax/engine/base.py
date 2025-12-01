"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import flax.struct # type: ignore
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
    static during JIT compilation, enabling optimal performance.
    
    Attributes:
        pop_size: Population size (must be positive)
        elitism: Number of elite individuals to preserve (0 <= elitism < pop_size)
        num_generations: Total generations to evolve (must be positive)
        
    Example:
        >>> params = AbstractEngineParams(
        ...     pop_size=100,
        ...     elitism=2,
        ...     num_generations=50
        ... )
    """
    pop_size: int = flax.struct.field(pytree_node=False)
    elitism: int = flax.struct.field(pytree_node=False)
    num_generations: int = flax.struct.field(pytree_node=False)
    
    # Note: __post_init__ validation removed to ensure JIT compatibility
    # Validation should be done at engine construction time instead


@flax.struct.dataclass
class AbstractEvolutionState(Generic[G, P]):
    """
    Mutable state container that evolves across generations.
    
    This class holds the dynamic state of the evolutionary process,
    including the population, best individual, and RNG state. Must contain only
    JAX-compatible types for JIT compilation with jax.lax.scan.
    
    Type Parameters:
        G: Genome type (subclass of BaseGenome)
        P: Population type (subclass of BasePopulation[G])
    
    Attributes:
        population: Current population of genomes
        best_genome: Best genome found so far
        generation: Current generation number (0-indexed)
        best_fitness: Best fitness value found so far
        stagnation_counter: Generations without fitness improvement
        rng_key: JAX PRNG key for reproducible randomness
        
    Example:
        >>> state = AbstractEvolutionState(
        ...     population=initial_pop,
        ...     best_genome=best_individual,
        ...     generation=0,
        ...     best_fitness=jnp.array(0.0),
        ...     stagnation_counter=0,
        ...     rng_key=jar.PRNGKey(42)
        ... )
    """
    # --- CRITICAL: Population and Best Individual ---
    population: P  # Current population
    best_genome: G  # Best genome found so far
    
    # --- EXISTING: Metadata ---
    generation: int
    best_fitness: jax.Array
    stagnation_counter: int
    rng_key: jax.Array
    fitness_values: jax.Array  

@flax.struct.dataclass
class AbstractGenerationOutput:
    """
    Base KPI payload returned at every evolution step.
    Foundation for universal dashboard generation.
    
    All subclasses must be immutable flax.struct.dataclass types
    containing only JAX arrays for JIT compatibility.
    
    Attributes:
        best_fitness: Scalar array with best fitness in generation
        mean_fitness: Scalar array with population mean fitness
        generation: Scalar integer array with generation number
        
    Example:
        >>> output = AbstractGenerationOutput(
        ...     best_fitness=jnp.array(0.95),
        ...     mean_fitness=jnp.array(0.78),
        ...     generation=jnp.array(42)
        ... )
        >>> kpis = output.get_kpi_names()
        >>> assert 'best_fitness' in kpis
    """
    best_fitness: jax.Array
    mean_fitness: jax.Array
    generation: jax.Array
    
    @classmethod
    def get_kpi_names(cls) -> List[str]:
        """
        Return available KPI field names for visualization.
        
        Returns:
            List of field names that can be extracted via get_kpi_value()
            
        Example:
            >>> names = AbstractGenerationOutput.get_kpi_names()
            >>> assert names == ['best_fitness', 'mean_fitness', 'generation']
        """
        return list(cls.__dataclass_fields__.keys())
    
    def get_kpi_value(self, kpi_name: str) -> jax.Array:
        """
        Extract specific KPI value by name.
        
        Args:
            kpi_name: Name of KPI field to extract
            
        Returns:
            JAX array containing the KPI value
            
        Raises:
            AttributeError: If kpi_name does not exist in this output type
            
        Example:
            >>> output = AbstractGenerationOutput(...)
            >>> best = output.get_kpi_value('best_fitness')
            >>> assert isinstance(best, jax.Array)
        """
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
    
    Provides standardized evolution loop with JAX scan optimization,
    optional JIT compilation, and comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize engine with compilation cache."""
        self._compiled_evolution_fn: Optional[Callable] = None
        self._compiled_for_params: Optional[AbstractEngineParams] = None
    
    @abstractmethod
    def init_state(self, rng_key: jnp.ndarray, params: AbstractEngineParams) -> AbstractEvolutionState:
        """
        Initialize the evolution state.
        
        Args:
            rng_key: JAX PRNG key for random initialization
            params: Engine configuration parameters
            
        Returns:
            Initial evolution state with population and metadata
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
        
        Args:
            key: JAX PRNG key for this generation
            state: Current evolution state
            params: Engine configuration parameters
            
        Returns:
            Tuple of (population, updated_state, generation_metrics)
        """
        pass
    
    def compile_evolution(self, params: AbstractEngineParams) -> None:
        """
        Pre-compile the evolution function for given parameters.
        
        This method compiles the evolution loop once and caches it for subsequent
        runs with the same parameters. This eliminates recompilation overhead
        when running multiple evolution experiments.
        
        Args:
            params: Engine parameters that define the compilation context
            
        Example:
            >>> engine = MyEngine(...)
            >>> params = AbstractEngineParams(pop_size=100, num_generations=50, elitism=2)
            >>> engine.compile_evolution(params)  # Compile once
            >>> # Now all runs with these params will use cached compilation
            >>> for seed in range(10):
            ...     state = engine.init_state(jar.PRNGKey(seed), params)
            ...     final_state, history, _ = engine.run(state, params)
        """
        validate_engine_params(params)
        
        # Bake static parameters into step function for JIT compatibility
        step_fn = functools.partial(self.step, params=params)
        
        def scan_body(
            carry: Tuple[jnp.ndarray, AbstractEvolutionState], 
            _
        ) -> Tuple[Tuple[jnp.ndarray, AbstractEvolutionState], AbstractGenerationOutput]:
            """Inner scan function - must be pure for JIT compilation."""
            rng_key, state = carry
            step_key, new_rng_key = jar.split(rng_key)
            
            # Execute one generation
            _, new_state, history_item = step_fn(step_key, state)
            
            return (new_rng_key, new_state), history_item
        
        # Define and compile the complete scan operation
        def _run_evolution(init_carry):
            return jax.lax.scan(
                scan_body, 
                init_carry, 
                None, 
                length=params.num_generations
            )
        
        # Compile and cache the evolution function
        # Use object.__setattr__ for frozen Flax dataclasses
        object.__setattr__(self, '_compiled_evolution_fn', jax.jit(_run_evolution))
        object.__setattr__(self, '_compiled_for_params', params)
        
        # Note: Warmup removed to avoid state type mismatches
        # The function will be compiled on first actual use with correct state type
    
    def is_compiled(self, params: Optional[AbstractEngineParams] = None) -> bool:
        """
        Check if evolution function is compiled for given parameters.
        
        Args:
            params: Parameters to check compilation for. If None, checks
                   if any compilation exists.
                   
        Returns:
            True if compiled function exists and matches params (if provided)
            
        Example:
            >>> engine = MyEngine(...)
            >>> assert not engine.is_compiled()
            >>> engine.compile_evolution(params)
            >>> assert engine.is_compiled(params)
        """
        if self._compiled_evolution_fn is None:
            return False
        
        if params is None:
            return True
            
        return (self._compiled_for_params is not None and 
                self._compiled_for_params == params)
    
    def clear_compilation_cache(self) -> None:
        """
        Clear cached compiled function.
        
        Use this when you need to free memory or when switching to
        significantly different parameter configurations.
        
        Example:
            >>> engine = MyEngine(...)
            >>> engine.compile_evolution(params1)
            >>> # ... run experiments ...
            >>> engine.clear_compilation_cache()  # Free memory
            >>> engine.compile_evolution(params2)  # Compile for new params
        """
        # Use object.__setattr__ for frozen Flax dataclasses
        object.__setattr__(self, '_compiled_evolution_fn', None)
        object.__setattr__(self, '_compiled_for_params', None)
    
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
        
        This method orchestrates the full evolutionary loop with:
        - Automatic RNG key management
        - Intelligent compilation caching (compiles once per parameter set)
        - Generation-wise KPI tracking
        - Timing and progress monitoring
        
        The method automatically manages JIT compilation:
        - If compile=True and function is cached for these params: uses cached version
        - If compile=True and not cached: compiles, caches, then runs
        - If compile=False: runs without JIT compilation
        
        Args:
            initial_state: Initial evolution state with population and RNG key
            params: Engine parameters (pop_size, num_generations, etc.)
            time_it: If True, measure and return execution time
            compile: If True, use JIT compilation with caching (recommended for production)
            verbose: If True, print progress, compilation status, and timing information
            
        Returns:
            Tuple of (final_state, history, elapsed_time)
            - final_state: Evolution state after all generations
            - history: AbstractGenerationOutput pytree with shape (num_generations,)
            - elapsed_time: Wall-clock time in seconds (None if time_it=False)
            
        Raises:
            ValueError: If params.num_generations <= 0
            TypeError: If initial_state or params have incorrect types
            
        Example:
            >>> engine = MyEngine(genome_config, fitness_fn, operators)
            >>> params = AbstractEngineParams(pop_size=100, num_generations=50, elitism=2)
            >>> 
            >>> # Option 1: Auto-compilation on first run
            >>> state = engine.init_state(jar.PRNGKey(42), params)
            >>> final_state, history, time = engine.run(state, params, compile=True)
            >>> 
            >>> # Option 2: Pre-compile for multiple runs (recommended for experiments)
            >>> engine.compile_evolution(params)  # Compile once
            >>> for seed in range(10):  # Multiple runs use cached compilation
            ...     state = engine.init_state(jar.PRNGKey(seed), params)
            ...     final_state, history, _ = engine.run(state, params, compile=True)
        """
        # Input validation
        if params.num_generations <= 0:
            raise ValueError(
                f"num_generations must be positive, got {params.num_generations}"
            )
        
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
        
        # Execute evolution loop
        start_time = time.time() if time_it else None
        init_carry = (initial_state.rng_key, initial_state)
        
        # Choose execution strategy based on compilation preference
        if compile:
            # Check if we have a cached compiled function for these parameters
            if self.is_compiled(params):
                if verbose:
                    print("Using cached compiled function")
                evolution_fn = self._compiled_evolution_fn
            else:
                if verbose:
                    print("Compiling evolution function (first run with these parameters)")
                # Compile and cache the function
                self.compile_evolution(params)
                evolution_fn = self._compiled_evolution_fn
        else:
            # Non-compiled execution - build function on the fly
            if verbose:
                print("Running without JIT compilation")
            
            # Bake static parameters into step function
            step_fn = functools.partial(self.step, params=params)
            
            def scan_body(
                carry: Tuple[jnp.ndarray, AbstractEvolutionState], 
                _
            ) -> Tuple[Tuple[jnp.ndarray, AbstractEvolutionState], AbstractGenerationOutput]:
                """Inner scan function - must be pure for JIT compilation."""
                rng_key, state = carry
                step_key, new_rng_key = jar.split(rng_key)
                
                # Execute one generation
                _, new_state, history_item = step_fn(step_key, state)
                
                return (new_rng_key, new_state), history_item
            
            # Define the complete scan operation without JIT
            def evolution_fn(init_carry):
                return jax.lax.scan(
                    scan_body, 
                    init_carry, 
                    None, 
                    length=params.num_generations
                )
        
        try:
            (final_key, final_state), history = evolution_fn(init_carry)
        except Exception as e:
            raise RuntimeError(
                f"Evolution loop failed at generation {initial_state.generation}: {str(e)}"
            ) from e
        
        # Synchronize and measure timing
        if time_it:
            jax.block_until_ready(final_state)
            elapsed_time = time.time() - start_time
            if verbose:
                gen_time = elapsed_time / params.num_generations
                print(f"Evolution completed in {elapsed_time:.3f}s "
                      f"({gen_time*1000:.2f}ms/gen)")
        else:
            elapsed_time = None
        
        # Update final state with correct RNG key
        final_state = final_state.replace(rng_key=final_key)
        
        if verbose:
            print(f"Final generation: {final_state.generation}")
            print(f"Best fitness: {float(final_state.best_fitness):.6f}")
            print(f"Stagnation counter: {final_state.stagnation_counter}")
        
        return final_state, history, elapsed_time