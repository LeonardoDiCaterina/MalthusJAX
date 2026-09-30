"""StubEngine backend provider — used when no real backend is selected."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


class StubProvider:
    """Fallback provider that returns a StubEngine for pipeline testing."""

    @property
    def name(self) -> str:
        return "stub"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        return BaseStrategy()

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        return False

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        return None

    def build_engine(
        self,
        strategy: BaseStrategy,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Any:
        from malthusjax.composer.factory import build_stub_engine

        return build_stub_engine(
            generations=generations,
            base_fitness=kwargs.get("base_fitness", 1.0),
            improvement_rate=kwargs.get("improvement_rate", 0.1),
            **kwargs,
        )

    def generate_initial_population(
        self,
        config: Dict[str, Any],
        pop_seed: int,
    ) -> Optional[Any]:
        return None


_stub_provider = StubProvider()
register_backend("stub", _stub_provider)
