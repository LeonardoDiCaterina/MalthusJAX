from typing import Any, Optional
from flax import struct
from malthusjax.composer.strategies.base import BaseStrategy
from malthusjax.operators.base import BaseSelection, BaseCrossover, BaseMutation
from malthusjax.operators.emitters.base import BaseEmitter

@struct.dataclass
class GeneticStrategy(BaseStrategy):
    """
    Standard Generation Engine (GeneticFastEngine) with genetic operators.
    If multiple operators are passed, they are automatically composed via a MixingEmitter.
    """
    selection: Optional[Any] = struct.field(pytree_node=False, default=None)
    crossover: Optional[Any] = struct.field(pytree_node=False, default=None)
    mutation: Optional[Any] = struct.field(pytree_node=False, default=None)

@struct.dataclass
class MapElitesStrategy(BaseStrategy):
    """
    MAP-Elites Engine. Expects an emitter (often a MixingEmitter).
    """
    emitter: Optional[BaseEmitter] = struct.field(pytree_node=False, default=None)

@struct.dataclass
class EvoSAXStrategy(BaseStrategy):
    """
    Wraps an EvoSAX algorithm.
    """
    algorithm_name: str = struct.field(pytree_node=False)
    algorithm_kwargs: dict = struct.field(pytree_node=False, default_factory=dict)

@struct.dataclass
class QDAXStrategy(BaseStrategy):
    """
    Wraps a QDAX algorithm.
    """
    strategy_cls: Any = struct.field(pytree_node=False)
    emitter: Any = struct.field(pytree_node=False)
    metrics_function: Any = struct.field(pytree_node=False)
    centroids: Any = struct.field(pytree_node=False)
    init_variables: Any = struct.field(pytree_node=False)
    algorithm_kwargs: dict = struct.field(pytree_node=False, default_factory=dict)
