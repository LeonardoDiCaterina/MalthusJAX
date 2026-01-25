from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: Dict[str, Any]

    def build(self, key: Any, registry: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Resolve factory and call it (kept simple and side-effect-free)."""
        factory = registry.get(self.type)
        return factory(key, self.params, inputs)
