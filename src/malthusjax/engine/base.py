"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""
import jax
import chex
import jax.numpy as jnp 
import flax.struct 
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Generic, TypeVar
import functools
import time
from flax import struct

# Import for generic typing - ensure these exist in your project
from malthusjax.core.base import BaseGenome, BasePopulation

# Type variables for generics
G = TypeVar('G', bound=BaseGenome)        # Genome type
P = TypeVar('P', bound=BasePopulation)    # Population type


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
    pop_size: int = flax.struct.field(pytree_node=False, default=100)
    elitism: int = flax.struct.field(pytree_node=False, default=0)
    num_generations: int = flax.struct.field(pytree_node=False, default=50)
    unroll_num: int = flax.struct.field(pytree_node=False, default=1)
    
    def __post_init__(self):
        
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
class AbstractEngine(ABC):
    """
    Abstract base class for all evolutionary engines.
    
    Standardizes the evolution loop using JAX scan.
    """
    engine_params: AbstractEngineParams = struct.field(pytree_node=False)
    
    @abstractmethod
    def init_state(self, rng_key: jnp.ndarray) -> AbstractEvolutionState:
        """
        Initialize the evolution state (Compile Plan & Bake Operators).
        Note: We removed 'params' from arg list because self.engine_params exists.
        """
        # Input validation should happen here or in constructor
        validate_engine_params(self.engine_params)
    
    @abstractmethod
    def step(
        self, 
        state: AbstractEvolutionState
    ) -> Tuple[AbstractEvolutionState, AbstractGenerationOutput]:
        """
        Execute one generation step.
        
        Args:
            state: Current state (containing rng_key, operators, population).
            
        Returns:
            (new_state, history_item)
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
        """
        if verbose:
            print(f"Starting evolution: {self.engine_params.num_generations} generations, "
                  f"population size {self.engine_params.pop_size}, compile={compile}")

        # Retrieve the compiled loop function
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
        initial_state: AbstractEvolutionState,
        optimize: bool = True,  # <--- NEW FLAG
        print_analysis: bool = True
    ) -> str:
        """
        Extracts HLO text. 
        If optimize=True, runs the full XLA compiler to show FUSION.
        If optimize=False, shows the raw graph (faster, good for debugging shapes).
        """
        print(f"--- Extracting HLO (Optimize={optimize}) ---")
        
        # 1. Get JIT Kernel
        jit_kernel = _get_evolution_kernel(self.engine_params, compile_jit=True, unroll_num=self.engine_params.unroll_num)
        
        # 2. Lower (Trace to StableHLO)
        lowered = jit_kernel.lower(self, initial_state)
        
        # 3. Compile (Run XLA Optimizations)
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
            
            print(f"HLO Analysis:")
            print(f"  - Total Lines of IR: {line_count}")
            print(f"  - Fusion Kernels:    {fusion_count} (Higher is better)")
            print(f"  - Explicit Loops:    {loop_count} (Should be 1 for the main scan)")
            print("-" * 30)
            
        return hlo_text

@functools.lru_cache(maxsize=32)
def _get_evolution_kernel(params: AbstractEngineParams, compile_jit: bool = True, unroll_num: int = 1):
    """
    Factory that builds and compiles the evolution loop.
    Cached by 'params' to ensure we only compile once per configuration.
    """
    
    # 1. Define the scan body
    # params is captured via closure
    def _scan_body(carry, _):
        engine, state = carry
        
        # Execute Step
        # We cleaned up the signature: engine.step(state)
        # The engine instance (self) is passed via carry to support polymorphism
        new_state, history_item = engine.step(state)
        
        # Scan requires: (carry, accum)
        return (engine, new_state), history_item

    # 2. Define the outer loop
    def _evolve_loop(engine: AbstractEngine, initial_state: AbstractEvolutionState):
        
        init_carry = (engine, initial_state)
        
        (final_engine, final_state), history = jax.lax.scan(
            _scan_body,
            init_carry,
            None, # Scan over range(num_generations) implicitly
            length=params.num_generations,
            unroll= unroll_num
        )
        
        return final_state, history

    # 3. JIT Compile
    if compile_jit:
        # donate_argnums=1: Donate 'initial_state' memory.
        # We DO NOT donate arg 0 (engine) because it's static/structural.
        return jax.jit(_evolve_loop, donate_argnums=1)
    else:
        return _evolve_loop