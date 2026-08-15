"""Elementary component of a Composer pipeline.

Nodes are frozen dataclasses containing an identifier, a type string that
resolves via a :class:`~.registry.Registry`, and a dictionary of parameters
passed to the factory.  The ``build()`` method executes the factory with a
supplied PRNG key and any upstream outputs.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: Dict[str, Any]

    def build(self, key: Any, registry: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke the registered factory to produce node output.

        The factory is responsible for interpreting *params* and combining any
        values from *inputs*; this class deliberately avoids embedding any
        logic beyond lookup and call dispatch.
        """
        factory = registry.get(self.type)
        return factory(key, self.params, inputs)
