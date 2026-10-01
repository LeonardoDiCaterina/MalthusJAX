"""MalthusJAX engine factory functions.

Core builder logic for native MalthusJAX engines and data registries.
Backend-specific builder functions have migrated into their respective
BackendProvider modules in ``malthusjax.composer.backends.*``.
"""

from typing import Any, Dict, Optional

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.engine_catalog import EngineRegistry
from malthusjax.composer.strategies.base import BaseStrategy
from malthusjax.composer.strategies.core import GeneticStrategy


def has_real_operators(
    genome: Optional[str],
    fitness: Optional[Any],
    selection: Optional[str],
    crossover: Optional[str],
    mutation: Optional[str],
) -> bool:
    """Check if any real operator specs are provided."""
    return any([genome, fitness, selection, crossover, mutation])


def build_data_registry(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a configuration of data sources into a resolved data registry."""
    from malthusjax.benchmarking.registry import DataRegistry

    reg = DataRegistry()
    for data_id, data_spec in data_config.items():
        reg.register(data_id, data_spec)
    resolved: Dict[str, Any] = {}
    for data_id in data_config.keys():
        resolved[data_id] = reg.resolve(data_id)
    return resolved


def _check_compatibility(operator: Any, genome_type: str, engine_type: str, role: str) -> None:
    """Validate operator against compatibility metadata if present."""
    if hasattr(operator, "_malthusjax_metadata"):
        metadata = operator._malthusjax_metadata
        comp_genomes = metadata.get("compatible_genomes")
        if comp_genomes is not None and genome_type not in comp_genomes:
            raise ValueError(
                f"Error: {role} is only compatible with genomes: {comp_genomes}, but you requested '{genome_type}'."
            )
        comp_engines = metadata.get("compatible_engines")
        if comp_engines is not None and engine_type not in comp_engines:
            raise ValueError(
                f"Error: {role} is only compatible with engines: {comp_engines}, but you requested '{engine_type}'."
            )


def build_real_engine(
    strategy: BaseStrategy,
    fitness: Optional[Any],
    engine_type: str = "ga",
    genome_type: str = "real",
    data_config: Optional[Dict[str, Any]] = None,
    **config: Any,
) -> Any:
    """Build engine from operator specs and config via EngineRegistry."""
    from malthusjax.engine.schedules import TrackBest

    catalog = OperatorCatalog()
    data_registry = None
    if data_config:
        data_registry = build_data_registry(data_config)
    seed_val = config.get("seed", 42)
    maximize_flag = config.get("maximize", False)
    # We append seed to fitness strings if missing so BBOB etc uses the right seed
    if isinstance(fitness, str):
        if "seed=" not in fitness:
            if ":" in fitness:
                fitness = f"{fitness},seed={seed_val}"
            else:
                fitness = f"{fitness}:seed={seed_val}"
        resolved_evaluator = catalog.get(
            fitness,
            data_registry=data_registry,
            maximize=maximize_flag,
        )
    elif isinstance(fitness, dict):
        if "seed" not in fitness:
            fitness["seed"] = seed_val
        resolved_evaluator = catalog.get(
            fitness,
            data_registry=data_registry,
            maximize=maximize_flag,
        )
    elif fitness is not None:
        # Assuming it's already an instantiated Evaluator object (from parse_evaluator)
        resolved_evaluator = fitness
    else:
        resolved_evaluator = catalog.get(
            f"sphere:dim=10,maximize={maximize_flag},seed={seed_val}",
            data_registry=data_registry,
        )
    if isinstance(strategy, GeneticStrategy):
        resolved_selection = (
            catalog.get(
                strategy.selection
                or f"tournament:num_selections={config.get('pop_size', 50) // 2},tournament_size=3",
                data_registry=data_registry,
            )
            if isinstance(strategy.selection, (str, dict)) or strategy.selection is None
            else strategy.selection
        )
        resolved_crossover = (
            catalog.get(strategy.crossover or "blend:alpha=0.5", data_registry=data_registry)
            if isinstance(strategy.crossover, (str, dict)) or strategy.crossover is None
            else strategy.crossover
        )
        resolved_mutation = (
            catalog.get(
                strategy.mutation or "gaussian:mutation_rate=0.1", data_registry=data_registry
            )
            if isinstance(strategy.mutation, (str, dict)) or strategy.mutation is None
            else strategy.mutation
        )

        # Enforce compatibility guardrails
        if resolved_selection:
            _check_compatibility(resolved_selection, genome_type, engine_type, "selection")
        if resolved_crossover:
            _check_compatibility(resolved_crossover, genome_type, engine_type, "crossover")
        if resolved_mutation:
            _check_compatibility(resolved_mutation, genome_type, engine_type, "mutation")
    else:
        resolved_selection = None
        resolved_crossover = None
        resolved_mutation = None
    # Ensure we use LIGHT tracking for monotonic convergence curves
    if "track_best" not in config:
        config["track_best"] = TrackBest.LIGHT
    engine_registry = EngineRegistry()
    return engine_registry.get(
        engine_type,
        evaluator=resolved_evaluator,
        selection=resolved_selection,
        crossover=resolved_crossover,
        mutation=resolved_mutation,
        **config,
    )
