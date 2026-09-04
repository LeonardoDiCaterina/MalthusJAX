"""Genome catalog — string-spec → genome config resolution.

The catalog allows creating genome configurations from string specifications.

Examples::

    catalog = GenomeCatalog()
    catalog.get("real:dim=10,bounds=(-5,5)")
    catalog.get("binary:length=20")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from ..core.genome.binary_genome import BinaryGenomeConfig
from ..core.genome.real_genome import RealGenomeConfig
from ..core.genome.linear_genome import LinearGenomeConfig
from ..core.genome.cartesian_genome import CartesianGenomeConfig
from ._genome_registry import get_registry, register_table
from ._genome_registry import register as _registry_register


def _register_builtins() -> None:
    def _create_real(
        dim: int = 10,
        bounds: str | Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> RealGenomeConfig:
        if isinstance(bounds, str):
            # Parse "(lower,upper)"
            b_str = bounds.strip("()[]")
            parts = b_str.split(",")
            bounds = (float(parts[0]), float(parts[1]))

        shape = kwargs.pop("shape", (dim,))
        return RealGenomeConfig(shape=shape, bounds=bounds, **kwargs)

    def _create_binary(length: int = 10, **kwargs: Any) -> BinaryGenomeConfig:
        shape = kwargs.pop("shape", (length,))
        return BinaryGenomeConfig(shape=shape, **kwargs)

    def _create_linear(**kwargs: Any) -> LinearGenomeConfig:
        return LinearGenomeConfig(**kwargs)

    def _create_cartesian(**kwargs: Any) -> CartesianGenomeConfig:
        return CartesianGenomeConfig(**kwargs)

    register_table(
        [
            ("real", _create_real, {"dim": 10, "bounds": (-5.0, 5.0)}),
            ("binary", _create_binary, {"length": 10}),
            ("linear", _create_linear, {"length": 10, "num_inputs": 2, "num_ops": 5, "max_arity": 2}),
            ("cartesian", _create_cartesian, {"num_rows": 1, "num_cols": 10, "levels_back": 10, "num_inputs": 2, "num_outputs": 1, "num_ops": 5, "max_arity": 2}),
        ],
        override=True,
    )


class GenomeCatalog:
    """Catalog for creating genomes from string specifications.

    Supports format: ``"genome_type:param1=value1,param2=value2"``

    Available Genomes:
        - real: Real-valued genome
        - binary: Binary genome
    """

    def __init__(self) -> None:
        _register_builtins()
        self._registry = get_registry()

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        spec = spec.strip()
        if not spec:
            raise ValueError("Empty genome specification")

        if ":" not in spec:
            return spec, {}

        genome_name, params_str = spec.split(":", 1)
        genome_name = genome_name.strip()
        params: Dict[str, Any] = {}

        if params_str.strip():
            parts: List[str] = []
            current: List[str] = []
            in_parens = 0
            for char in params_str:
                if char == "(":
                    in_parens += 1
                elif char == ")":
                    in_parens -= 1

                if char == "," and in_parens == 0:
                    parts.append("".join(current))
                    current = []
                else:
                    current.append(char)
            if current:
                parts.append("".join(current))

            for param_pair in parts:
                param_pair = param_pair.strip()
                if "=" not in param_pair:
                    continue  # skip invalid
                key, value = param_pair.split("=", 1)
                params[key.strip()] = self._convert_value(value.strip())

        return genome_name, params

    @staticmethod
    def _convert_value(value_str: str) -> Union[int, float, str, bool]:
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        if value_str.startswith("(") and value_str.endswith(")"):
            return value_str

        try:
            return int(value_str)
        except ValueError:
            pass

        try:
            return float(value_str)
        except ValueError:
            pass

        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        return value_str

    def get(self, spec: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
        if isinstance(spec, dict):
            spec_dict = spec.copy()
            if "type" not in spec_dict:
                raise ValueError("Genome specification dict must contain a 'type' key.")
            genome_name = spec_dict.pop("type")
            spec_params = spec_dict
        else:
            genome_name, spec_params = self.parse_spec(spec)

        if genome_name not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(f"Unknown genome '{genome_name}'. Available: [{available}]")

        factory, defaults, metadata = self._registry[genome_name]
        merged_params = {**defaults, **spec_params, **kwargs}

        try:
            genome_config = factory(**merged_params)
            if not hasattr(genome_config, "_malthusjax_metadata"):
                try:
                    object.__setattr__(genome_config, "_malthusjax_metadata", metadata)
                except (TypeError, AttributeError):
                    pass
            return genome_config
        except TypeError as e:
            raise ValueError(f"Invalid parameters for genome '{genome_name}': {e}") from e

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        defaults: Dict[str, Any] | None = None,
        override: bool = False,
    ) -> None:
        if not override and name in self._registry:
            raise KeyError(f"Genome '{name}' is already registered")

        _registry_register(name, factory, defaults, override=True)
        self._registry[name] = (factory, defaults or {}, {})

    def list_available(self) -> List[str]:
        return sorted(self._registry.keys())
