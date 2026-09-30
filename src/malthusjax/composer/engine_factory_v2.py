"""EngineFactory — the single config -> engine boundary.

Replaces inline engine construction blocks in quick_run() and compare()
with a unified instantiation boundary. Registry lookup, evaluator resolution,
strategy construction, and engine creation all happen here.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .backend_registry import get_backends, list_backends
from .evaluator_parser import parse_evaluator
from .experiment_config import (
    EvosaxBackendConfig,
    ExperimentConfig,
    MalthusJAXBackendConfig,
    QdaxBackendConfig,
    TensorneatBackendConfig,
)


class EngineFactory:
    """Builds Engine instances from ExperimentConfig.

    This is the Hydra-style instantiate() for MalthusJAX.
    """

    def build(self, config: ExperimentConfig) -> Any:
        """Build and return an Engine conforming to the Engine protocol.

        Parameters
        ----------
        config : ExperimentConfig
            Complete experiment configuration.

        Returns
        -------
        Engine
            Instantiated engine ready for execution.
        """
        engine, _ = self.build_with_evaluator(config)
        return engine

    def build_with_evaluator(self, config: ExperimentConfig) -> Tuple[Any, Any]:
        """Build engine and return both the engine and its resolved evaluator.

        Parameters
        ----------
        config : ExperimentConfig
            Complete experiment configuration.

        Returns
        -------
        Tuple[Engine, Any]
            (engine, evaluator) pair.
        """
        if not isinstance(config, ExperimentConfig):
            if isinstance(config, dict):
                config = ExperimentConfig.from_dict(config)
            else:
                raise TypeError(f"Expected ExperimentConfig, got {type(config).__name__}")

        # Validate config before instantiating
        config.validate()

        backend_name = config.backend.name.lower()
        backends = get_backends()

        if backend_name not in backends:
            from .plugins import load_plugins

            load_plugins()
            backends = get_backends()

        if backend_name not in backends:
            available = ", ".join(list_backends())
            raise ValueError(f"Unknown backend '{backend_name}'. Available: [{available}]")

        provider_entry = backends[backend_name]
        provider = provider_entry[0]
        provider_defaults = provider_entry[1]

        # Resolve raw fitness spec / dict if present
        fitness_raw = getattr(config.backend, "fitness", None)
        if isinstance(fitness_raw, dict):
            fitness_obj = parse_evaluator(fitness_raw)
        else:
            fitness_obj = fitness_raw

        # Infer dimensions from fitness spec if still default and available
        genome_length = config.population.genome_length
        if genome_length == 10 and isinstance(fitness_raw, str):
            from .catalog import OperatorCatalog

            try:
                _, parsed_params = OperatorCatalog().parse_spec(fitness_raw)
                dim_val = parsed_params.get("dim", parsed_params.get("num_dims"))
                if dim_val is not None:
                    genome_length = int(dim_val)
            except Exception:
                pass

        # Build strategy kwargs
        strategy_kwargs: Dict[str, Any] = {**provider_defaults}
        if isinstance(config.backend, EvosaxBackendConfig):
            strategy_kwargs["evosax_strategy"] = config.backend.strategy
            strategy_kwargs["composable"] = config.backend.composable
            strategy_kwargs["algorithm_kwargs"] = config.backend.algorithm_kwargs
            strategy_kwargs.update(config.backend.extra_kwargs)
        elif isinstance(config.backend, QdaxBackendConfig):
            strategy_kwargs["qdax_strategy"] = config.backend.strategy_cls
            strategy_kwargs["qdax_num_descriptors"] = config.backend.num_descriptors
            strategy_kwargs["qdax_num_centroids"] = config.backend.num_centroids
            strategy_kwargs["qdax_mutation_sigma"] = config.backend.mutation_sigma
            strategy_kwargs["algorithm_kwargs"] = config.backend.algorithm_kwargs
            strategy_kwargs.update(config.backend.extra_kwargs)
        elif isinstance(config.backend, TensorneatBackendConfig):
            strategy_kwargs["tensorneat_algorithm"] = config.backend.algorithm
            strategy_kwargs["tensorneat_genome"] = config.backend.genome_name
            strategy_kwargs["tensorneat_problem"] = config.backend.problem
            strategy_kwargs["tensorneat_num_inputs"] = config.backend.num_inputs
            strategy_kwargs["tensorneat_num_outputs"] = config.backend.num_outputs
            strategy_kwargs["composable"] = config.backend.composable
            strategy_kwargs["algorithm_kwargs"] = config.backend.algorithm_kwargs
            strategy_kwargs.update(config.backend.extra_kwargs)
        elif isinstance(config.backend, MalthusJAXBackendConfig):
            strategy_kwargs["engine_type"] = config.backend.engine_type
            strategy_kwargs["genome"] = config.backend.genome or config.population.genome_spec
            strategy_kwargs["fitness"] = fitness_obj
            strategy_kwargs["selection"] = config.backend.selection
            strategy_kwargs["crossover"] = config.backend.crossover
            strategy_kwargs["mutation"] = config.backend.mutation
            strategy_kwargs["elitism"] = config.backend.elitism
            strategy_kwargs["data_config"] = config.backend.data_config
            strategy_kwargs.update(config.backend.extra_kwargs)
        else:
            strategy_kwargs.update(config.backend.extra_kwargs)
        strategy_obj = config.backend.extra_kwargs.get("strategy")
        if strategy_obj is not None:
            strategy = strategy_obj
        else:
            strategy = provider.default_strategy(**strategy_kwargs)

        # Resolve evaluator
        eval_seed = config.backend.extra_kwargs.get("seed", 42)
        eval_kwargs: Dict[str, Any] = {
            "maximize": config.execution.maximize,
            "seed": eval_seed,
            "num_dims": genome_length,
            "bounds": config.population.bounds,
            **provider_defaults,
            **config.backend.extra_kwargs,
        }
        if isinstance(config.backend, MalthusJAXBackendConfig):
            eval_kwargs["data_config"] = config.backend.data_config

        evaluator = provider.resolve_evaluator(fitness_obj, **eval_kwargs)

        # Build step logging if configured
        step_logging = None
        if config.logging.log_interval is not None:
            from ..core.logger import StepLoggingConfig

            step_logging = StepLoggingConfig(
                log_interval=config.logging.log_interval,
                log_nan_watchdog=config.logging.log_nan_watchdog,
                logger_name="malthusjax.composer",
            )

        # Build engine
        engine_kwargs: Dict[str, Any] = {
            "pop_size": config.population.size,
            "generations": config.execution.generations,
            "maximize": config.execution.maximize,
            "bounds": config.population.bounds,
            "genome_length": genome_length,
            "genome_type": config.population.genome_type,
            "prng_impl": config.execution.prng_impl,
            "history_metrics": list(config.logging.history_metrics)
            if config.logging.history_metrics
            else None,
            "step_logging": step_logging,
            **provider_defaults,
            **config.backend.extra_kwargs,
        }

        if isinstance(config.backend, MalthusJAXBackendConfig):
            engine_kwargs["engine_type"] = config.backend.engine_type
            engine_kwargs["elitism"] = config.backend.elitism
            engine_kwargs["genome"] = config.backend.genome or config.population.genome_spec
            engine_kwargs["data_config"] = config.backend.data_config
        elif isinstance(config.backend, EvosaxBackendConfig):
            engine_kwargs["composable"] = config.backend.composable
            engine_kwargs.update(config.backend.algorithm_kwargs)
        elif isinstance(config.backend, QdaxBackendConfig):
            engine_kwargs.update(config.backend.algorithm_kwargs)
        elif isinstance(config.backend, TensorneatBackendConfig):
            engine_kwargs["composable"] = config.backend.composable
            engine_kwargs.update(config.backend.algorithm_kwargs)

        engine_kwargs.pop("strategy", None)
        engine_kwargs.pop("evaluator", None)
        engine = provider.build_engine(strategy, evaluator, **engine_kwargs)

        if step_logging is not None and hasattr(engine, "step_logging"):
            engine.step_logging = step_logging

        return engine, evaluator
