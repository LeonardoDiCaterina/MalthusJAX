from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .node import Node


@dataclass
class Pipeline:
    name: str
    nodes: List[Node]
    wiring: Dict[str, Sequence[str]] = field(default_factory=dict)

    def validate(self, registry: Any) -> None:
        for node in self.nodes:
            # ensure node type exists
            registry.get(node.type)

    def build(self, master_key: Any, registry: Any) -> Dict[str, Any]:
        """Instantiate nodes sequentially; key-splitting left to future work."""
        results: Dict[str, Any] = {}
        for node in self.nodes:
            results[node.id] = node.build(master_key, registry, inputs=results)
        return results
