"""BackendProvider protocol — the contract every backend must satisfy.

Each backend module implements this protocol and self-registers via
``register_backend()`` at module level.  The Composer performs a single
registry lookup to dispatch engine construction, replacing the if/elif
cascades in ``composer.py`` and the duplicated ``build_*`` functions
in ``factory.py``.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable

from malthusjax.composer.engine_protocol import Engine
from malthusjax.composer.strategies.base import BaseStrategy


@runtime_checkable
class BackendProvider(Protocol):
    """Contract that each backend module must implement.

    Implementations self-register via ``register_backend()`` at module-level.
    The Composer performs a single registry lookup to dispatch engine
    construction, replacing the if/elif cascades.

    Methods
    -------
    ``name : str`` (property)
        Canonical backend name used as the registry key.
    ``default_strategy(**user_kwargs) -> BaseStrategy``
        Create the default Strategy dataclass from user-facing kwargs.
    ``resolve_evaluator(fitness_spec, ...) -> Any``
        Resolve a fitness spec into a concrete evaluator.
    ``build_engine(strategy, evaluator, ...) -> Any``
        Build and return a concrete engine adapter.
    ``generate_initial_population(config, pop_seed) -> Optional[Any]``
        Generate a shared initial population for compare() parity.
    """

    @property
    def name(self) -> str:
        """Canonical backend name (e.g. ``'evosax'``).

        Used as the primary registry key.
        """
        ...

    def default_strategy(self, **user_kwargs: Any) -> BaseStrategy:
        """Create the default Strategy dataclass from user-facing kwargs.

        Replaces ``composer.py`` Cascade 1 (L464-496): the backend-specific
        strategy auto-creation logic.

        Example::

            EvosaxProvider().default_strategy(evosax_strategy="CMA_ES")
            # -> EvoSAXStrategy(algorithm_name="CMA_ES")
        """
        ...

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        """Return True if this provider can handle the given strategy instance.

        Enables Composer to infer the backend when an explicit strategy object
        is passed without a matching backend name.
        """
        ...

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
        """Resolve a fitness spec (str, dict, or pre-built object) into a
        concrete evaluator.

        Replaces the 4× duplicated ``isinstance(fitness_spec, str)`` blocks
        in ``factory.py``.  Common logic lives in
        ``backends/_evaluator_resolver.resolve_evaluator_base()``;
        backend-specific overrides go here.
        """
        ...

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
    ) -> Engine:
        """Build and return a concrete engine adapter satisfying the Engine protocol."""
        ...

    def generate_initial_population(
        self,
        config: Any,
        pop_seed: int,
    ) -> Optional[Any]:
        """Generate a shared initial population for ``compare()`` parity.

        Replaces the backend-specific branches in ``composer.py`` L688-744.
        Returns ``None`` if the backend doesn't support shared initialization,
        in which case the Composer falls back to uniform random.
        """
        ...
