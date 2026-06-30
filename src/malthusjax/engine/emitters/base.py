from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import chex
from flax import struct

from malthusjax.core.base import BasePopulation

@struct.dataclass
class EmitterState:
    """Base class for any emitter-specific state (e.g. running means, generation counters)."""
    pass

class BaseEmitter(ABC):
    """
    Abstract Base Class for Native MalthusJAX Quality-Diversity Emitters.
    """
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The number of genomes generated per generation cycle."""
        pass

    @abstractmethod
    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> Optional[EmitterState]:
        """Initializes any required internal state using the initial population."""
        pass
        
    @abstractmethod
    def ask(self, state: Optional[EmitterState], repertoire: Any, key: chex.Array) -> Tuple[BasePopulation, Optional[EmitterState]]:
        """
        Samples parents from the repertoire and generates a batch of mutated offspring.
        
        Args:
            state: The current emitter state.
            repertoire: The QDAX MapElitesRepertoire (or similar) containing the archive of elites.
            key: PRNGKey for stochastic generation.
            
        Returns:
            A tuple of (generated_offspring_population, updated_emitter_state).
        """
        pass
        
    def tell(
        self, 
        state: Optional[EmitterState], 
        repertoire: Any,
        population: BasePopulation, 
        fitnesses: chex.Array, 
        descriptors: chex.Array,
        key: chex.Array
    ) -> Optional[EmitterState]:
        """
        Updates the internal Emitter state using the evaluated population metrics.
        (e.g., CMA-ES rank updates).
        """
        return state
