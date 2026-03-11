"""Simple representation of a directed computation graph.

A :class:`Pipeline` consists of named :class:`~.node.Node` objects and an
optional wiring table describing dependencies.  The class provides
validation against a :class:`~.registry.Registry` and a basic builder that
invokes each node in sequence.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .node import Node


@dataclass
class Pipeline:
    name: str
    nodes: List[Node]
    wiring: Dict[str, Sequence[str]] = field(default_factory=dict)

    def validate(self, registry: Any) -> None:
        """Check that every node refers to a registered factory.

        A ``KeyError`` will be raised by the registry if any node type is
        unknown, so callers can catch and report configuration errors early.
        """
        for node in self.nodes:
            # ensure node type exists
            registry.get(node.type)

    def build(self, master_key: Any, registry: Any) -> Dict[str, Any]:
        """Instantiate nodes sequentially; key-splitting left to future work."""
        results: Dict[str, Any] = {}
        for node in self.nodes:
            results[node.id] = node.build(master_key, registry, inputs=results)
        return results
