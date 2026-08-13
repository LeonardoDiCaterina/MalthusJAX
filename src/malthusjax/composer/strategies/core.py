from typing import Any, Optional

from flax import struct

from malthusjax.composer.strategies.base import BaseStrategy
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
    Native MAP-Elites Engine. Expects an emitter (often a MixingEmitter).
    """

    emitter: Optional[BaseEmitter] = struct.field(pytree_node=False, default=None)
    num_descriptors: int = struct.field(pytree_node=False, default=2)
    num_centroids: int = struct.field(pytree_node=False, default=100)
    mutation_sigma: float = struct.field(pytree_node=False, default=0.1)
    key_derivation: str = struct.field(pytree_node=False, default="fold_in")
    centroids: Any = struct.field(pytree_node=False, default=None)
    maximize: bool = struct.field(pytree_node=False, default=False)


@struct.dataclass
class EvoSAXStrategy(BaseStrategy):
    """
    Wraps an EvoSAX algorithm.
    """

    algorithm_name: str = struct.field(pytree_node=False)
    algorithm_kwargs: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


@struct.dataclass
class QDAXStrategy(BaseStrategy):
    """
    Wraps a QDAX algorithm.
    """

    strategy_cls: Any = struct.field(pytree_node=False, default="MAPElites")
    emitter: Any = struct.field(pytree_node=False, default=None)
    metrics_function: Any = struct.field(pytree_node=False, default=None)
    centroids: Any = struct.field(pytree_node=False, default=None)
    init_variables: Any = struct.field(pytree_node=False, default=None)
    num_descriptors: int = struct.field(pytree_node=False, default=2)
    num_centroids: int = struct.field(pytree_node=False, default=100)
    mutation_sigma: float = struct.field(pytree_node=False, default=0.1)
    algorithm_kwargs: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


@struct.dataclass
class TensorNEATStrategy(BaseStrategy):
    """
    Wraps a TensorNEAT NEAT/HyperNEAT algorithm.
    """

    algorithm_name: str = struct.field(pytree_node=False, default="NEAT")
    genome_name: str = struct.field(pytree_node=False, default="default")
    problem_name: Optional[str] = struct.field(pytree_node=False, default=None)
    num_inputs: int = struct.field(pytree_node=False, default=2)
    num_outputs: int = struct.field(pytree_node=False, default=1)
    algorithm_kwargs: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)
