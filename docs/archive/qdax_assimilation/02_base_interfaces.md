# Base Interface Extensions

To support QDax (and Quality-Diversity in general), MalthusJAX's core interfaces require non-breaking extensions. 

## 1. The Evaluator (`src/malthusjax/core/fitness/base.py`)
Standard GAs evaluate only `fitness`. QD algorithms evaluate `fitness` and behavioral `descriptors`.

**Action Item:** 
Extend `BaseEvaluator` to optionally return descriptors.
```python
class BaseEvaluator(abc.ABC):
    # Existing
    @abc.abstractmethod
    def evaluate(self, genomes: PyTree) -> jnp.ndarray:
        pass
        
    # New Extension
    def evaluate_qd(self, genomes: PyTree) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns a tuple of (fitnesses, descriptors).
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("This evaluator does not support Quality-Diversity descriptors.")
```

## 2. Emitter Base (`src/malthusjax/operators/emitters/base.py`)
MalthusJAX's current Operators (Mutation/Crossover) are stateless pure functions. QDax uses "Emitters" which maintain state (e.g., CMA-ES covariance matrices, Reinforcement Learning policy weights).

**Action Item:**
Create a new `BaseEmitter` protocol in MalthusJAX.
```python
class BaseEmitter(abc.ABC):
    @abc.abstractmethod
    def init_state(self, key: jnp.ndarray) -> PyTree:
        pass
        
    @abc.abstractmethod
    def emit(self, repertoire: PyTree, state: PyTree, key: jnp.ndarray) -> Tuple[PyTree, PyTree]:
        """Returns (new_genotypes, updated_emitter_state)"""
        pass
```

## 3. The Repertoire State Wrapper
QDax tracks elites in a spatial grid (`MapElitesRepertoire`). MalthusJAX engines track state via `EngineState`. 

**Action Item:**
Create a `QDaxEngineState` that inherits from MalthusJAX's `BaseEngineState` but wraps the QDax PyTrees.
```python
from flax.struct import PyTreeNode

class QDaxEngineState(BaseEngineState):
    repertoire: PyTreeNode      # The QDax Repertoire
    emitter_state: PyTreeNode   # The QDax EmitterState
    
    @property
    def best_fitness(self) -> jnp.ndarray:
        return jnp.max(self.repertoire.fitnesses)
```
