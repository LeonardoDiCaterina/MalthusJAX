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
from flax import struct
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

@struct.dataclass
class AbstractEngine(ABC):
    """
    Abstract base class for all evolutionary engines.
    
    Standardizes the evolution loop using JAX scan.
    Leverages JAX's internal compilation cache (memory & disk) automatically.
    """
    engine_params: AbstractEngineParams = struct.field(pytree_node=False)
    
    @abstractmethod
    def init_state(self, rng_key: jnp.ndarray, params: AbstractEngineParams) -> AbstractEvolutionState:
        """
        Initialize the evolution state.
        """
        # Input validation
        validate_engine_params(self.engine_params)
        
        if not isinstance(self.engine_params, AbstractEngineParams):
            raise TypeError(
                f"params must be AbstractEngineParams, got {type(self.engine_params)}"
            )
    
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
        if not isinstance(initial_state, AbstractEvolutionState):
            raise TypeError(
                f"initial_state must be AbstractEvolutionState, got {type(initial_state)}"
            )
        if verbose:
            print(f"Starting evolution: {self.engine_params.num_generations} generations, "
                  f"population size {self.engine_params.pop_size}, compile={compile}")

        evolve_fn = _get_evolution_kernel(self.engine_params, compile_jit=compile)

        start_time = time.time()
        
        try:
            final_state, history = evolve_fn(self, initial_state)        
            
            if time_it:
                jax.block_until_ready(final_state)
                
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
        initial_state: AbstractEvolutionState,
        print_analysis: bool = True
    ) -> str:
        """
        Extracts the compiled HLO (High Level Optimizer) text for analysis.
        
        This is useful for debugging:
        - Fusion: Check if ops are fused (fewer 'kernels' is better).
        - Memory: Check buffer allocation and donation.
        - Loop Unrolling: Verify if XLA unrolled the generation loop.
        
        Args:
            initial_state: A sample state (used for shape/type tracing only).
            params: The configuration to compile for.
            print_analysis: If True, prints a summary of the HLO stats.
            
        Returns:
            The raw HLO text string.
        """
        print(f"--- Lowering HLO for params: {self.engine_params} ---")
        
        # 1. Get the JIT-wrapped kernel
        # We enforce compile_jit=True to access the .lower() API
        jit_kernel = _get_evolution_kernel(self.engine_params, compile_jit=True)
        
        # 2. Lower the function (Trace + Convert to IR)
        # We must pass the exact same argument types as run(): (self, initial_state)
        lowered = jit_kernel.lower(self, initial_state)
        
        # 3. Compile (Optional here, but necessary to see post-optimization fusion)
        # Use .as_text() to get the HLO IR
        hlo_text = lowered.as_text()
        
        if print_analysis:
            # Simple heuristic analysis of the text
            #  - conceptually what we are parsing
            line_count = len(hlo_text.split('\n'))
            fusion_count = hlo_text.count("fusion")
            loop_count = hlo_text.count("while")
            
            print(f"HLO Analysis:")
            print(f"  - Total Lines of IR: {line_count}")
            print(f"  - Fusion Kernels:    {fusion_count} (Higher is usually better for GPU)")
            print(f"  - Explicit Loops:    {loop_count} (Should ideally be 1 for the main loop)")
            print(f"  - Input Donation:    {'donate_argnums' in str(jit_kernel)}")
            print("-" * 30)
            
        return hlo_text

@functools.lru_cache(maxsize=32)
def _get_evolution_kernel(params: AbstractEngineParams, compile_jit: bool = True):
    """
    Factory that builds and compiles the evolution loop.
    
    Because this is cached via lru_cache, the expensive logic inside 
    (tracing and compiling) only happens ONCE per unique 'params' configuration.
    
    Args:
        params: Static configuration (hashable key).
        compile_jit: Whether to wrap in jax.jit.
        
    Returns:
        A compiled function with signature (engine, initial_state) -> (final, history)
    """
    
    # 1. Define the scan body
    # This captures 'params' as a closure (static constant in the graph)
    def _scan_body(carry, _):
        engine, state = carry
        
        # Extract key from state (state contains the master key)
        rng_key = state.rng_key
        
        # Call the engine's step. 
        # Note: We pass 'engine' from the carry, preserving polymorphism.
        _, new_state, history_item = engine.step(rng_key, state)
        
        # Return (carry, accumulated_output)
        return (engine, new_state), history_item

    # 2. Define the outer loop
    def _evolve_loop(engine: AbstractEngine, initial_state: AbstractEvolutionState):
        
        # jax.lax.scan requires a carry. We bundle (engine, state) as carry.
        # JAX is smart enough to see 'engine' doesn't change and optimizes it out.
        init_carry = (engine, initial_state)
        
        (final_engine, final_state), history = jax.lax.scan(
            _scan_body,
            init_carry,
            None, # Scan over None (length driven)
            length=params.num_generations
        )
        
        return final_state, history

    # 3. JIT Compile
    if compile_jit:
        # donate_argnums=1: Donate 'initial_state' memory to 'final_state' 
        # to save VRAM. We do NOT donate arg 0 (engine).
        return jax.jit(_evolve_loop, donate_argnums=1)
    else:
        return _evolve_loop
    
    
    
#==============================================================================
# 4. PyTree Registration Helper
# ==============================================================================

# We must ensure AbstractEngine (and subclasses) are treated as PyTrees 
# so they can be passed through jax.jit and jax.lax.scan.
# This registers the base class and effectively any subclass that doesn't 
# override tree_flatten (unless they use flax.struct.dataclass, which handles this).

def _engine_flatten(v):
    # Flatten strategy: assume engine has no dynamic data (parameters are separate).
    # If subclasses have data, they must implement their own registration or
    # use @flax.struct.dataclass
    return (), None

def _engine_unflatten(aux, children):
    # This is tricky for an ABC. 
    # Realistically, subclasses should be dataclasses. 
    # For the abstract base, we provide a dummy impl to satisfy JAX registry 
    # if it's ever inspected, but typically 'aux' would contain the class type 
    # in a more complex setup.
    #
    # Best Practice: Subclasses of AbstractEngine MUST be valid PyTrees.
    return object.__new__(AbstractEngine) 

#jax.tree_util.register_pytree_node(AbstractEngine, _engine_flatten, _engine_unflatten)