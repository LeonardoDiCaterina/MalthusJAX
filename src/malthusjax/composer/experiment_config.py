"""Structured experiment configuration — the single source of truth.

Every parameter that was previously a loose kwarg on quick_run() lives
in a typed, validated, serializable config tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PopulationConfig:
    """Genome and population parameters."""

    size: int = 50
    genome_type: str = "real"  # "real" | "binary" | custom
    genome_length: int = 10
    bounds: Tuple[float, float] = (-5.0, 5.0)
    genome_spec: Optional[str] = None  # declarative genome spec, e.g. "real:dim=10"


@dataclass(frozen=True)
class ExecutionConfig:
    """Run control parameters."""

    generations: int = 100
    seeds: Tuple[int, ...] = (1, 2, 3)
    maximize: bool = False
    prng_impl: Optional[str] = None
    use_history_for_final: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and monitoring parameters."""

    log_level: Optional[str] = None
    log_interval: Optional[int] = None
    log_nan_watchdog: bool = True
    history_metrics: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class OutputConfig:
    """Output and tracing parameters."""

    experiment_name: str = "quick_experiment"
    output_dir: Optional[str] = None
    trace_dir: Optional[str] = None


# ── Backend configs (discriminated union) ──────────────────────────


@dataclass(frozen=True)
class BackendConfig:
    """Base for all backend configurations. Subclass per backend."""

    name: str = "malthusjax"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MalthusJAXBackendConfig(BackendConfig):
    """Native MalthusJAX GA engine configuration."""

    name: str = "malthusjax"
    fitness: Optional[str] = None
    selection: Optional[str] = None
    crossover: Optional[str] = None
    mutation: Optional[str] = None
    engine_type: str = "ga"
    elitism: int = 2
    genome: Optional[str] = None
    data_config: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EvosaxBackendConfig(BackendConfig):
    """EvoSAX strategy configuration."""

    name: str = "evosax"
    strategy: str = "SimpleGA"
    fitness: Optional[str] = None
    composable: bool = False
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QdaxBackendConfig(BackendConfig):
    """QDAX quality-diversity configuration."""

    name: str = "qdax"
    strategy_cls: str = "MAPElites"
    fitness: Optional[str] = None
    num_descriptors: int = 2
    num_centroids: int = 100
    mutation_sigma: float = 0.1
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorneatBackendConfig(BackendConfig):
    """TensorNEAT neuroevolution configuration."""

    name: str = "tensorneat"
    algorithm: str = "NEAT"
    genome_name: str = "DefaultGenome"
    problem: Optional[str] = None
    num_inputs: int = 2
    num_outputs: int = 1
    composable: bool = False
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StubBackendConfig(BackendConfig):
    """Stub engine configuration (for testing pipelines)."""

    name: str = "stub"


@dataclass(frozen=True)
class GenericBackendConfig(BackendConfig):
    """Generic or third-party plugin backend configuration."""

    name: str = "generic"


# ── Top-level config ───────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, serializable experiment specification.

    No runtime objects — purely data describing what to build and run.
    This is the single argument to EngineFactory.build().
    """

    population: PopulationConfig = field(default_factory=PopulationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backend: BackendConfig = field(default_factory=MalthusJAXBackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a nested dict compatible with JSON / TOML serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Deserialize from either a nested or a flat dict (TOML-compatible)."""
        if "population" in d and isinstance(d["population"], dict):
            pop_data = dict(d["population"])
            if "bounds" in pop_data and isinstance(pop_data["bounds"], (list, tuple)):
                pop_data["bounds"] = (float(pop_data["bounds"][0]), float(pop_data["bounds"][1]))
            population = PopulationConfig(**pop_data)

            exec_data = dict(d.get("execution", {}))
            if "seeds" in exec_data and isinstance(exec_data["seeds"], (list, tuple)):
                exec_data["seeds"] = tuple(int(s) for s in exec_data["seeds"])
            execution = ExecutionConfig(**exec_data)

            log_data = dict(d.get("logging", {}))
            if "history_metrics" in log_data and isinstance(
                log_data["history_metrics"], (list, tuple)
            ):
                log_data["history_metrics"] = tuple(log_data["history_metrics"])
            logging = LoggingConfig(**log_data)

            output = OutputConfig(**d.get("output", {}))

            b_data = dict(d.get("backend", {}))
            b_name = b_data.get("name", "malthusjax")
            backend: BackendConfig
            if b_name == "malthusjax":
                backend = MalthusJAXBackendConfig(**b_data)
            elif b_name in ("evosax", "composable_evosax"):
                backend = EvosaxBackendConfig(**b_data)
            elif b_name == "qdax":
                backend = QdaxBackendConfig(**b_data)
            elif b_name in ("tensorneat", "composable_tensorneat"):
                backend = TensorneatBackendConfig(**b_data)
            elif b_name == "stub":
                backend = StubBackendConfig(**b_data)
            else:
                backend = GenericBackendConfig(**b_data)

            return cls(
                population=population,
                execution=execution,
                backend=backend,
                logging=logging,
                output=output,
            )
        # Flat dict fallback — convert via from_quick_run_kwargs
        return cls.from_quick_run_kwargs(**d)

    @classmethod
    def from_quick_run_kwargs(cls, **kwargs: Any) -> "ExperimentConfig":
        """Build config from the flat quick_run() parameter namespace.

        This is the backward-compatibility bridge: quick_run() calls this
        to convert its loose parameters into a structured config.
        """
        # Resolve genome defaults and specification
        genome_spec = kwargs.get("genome")
        genome_type = kwargs.get("genome_type")
        genome_length = kwargs.get("genome_length")
        bounds = kwargs.get("bounds")
        fitness = kwargs.get("fitness")

        if genome_spec is not None:
            from .genome_catalog import GenomeCatalog

            cat = GenomeCatalog()
            if isinstance(genome_spec, dict):
                g_type = genome_spec.get("type", "real")
                g_params = genome_spec.copy()
            else:
                g_type, g_params = cat.parse_spec(str(genome_spec))
            if genome_type is None:
                genome_type = g_type
            if genome_length is None:
                genome_length = g_params.get("dim", g_params.get("length", 10))
            if bounds is None and "bounds" in g_params:
                b = g_params["bounds"]
                if isinstance(b, str):
                    b_str = b.strip("()[]")
                    parts = b_str.split(",")
                    bounds = (float(parts[0]), float(parts[1]))
                else:
                    bounds = tuple(b)
            if genome_length is None and "shape" in g_params:
                shape_val = g_params["shape"]
                genome_length = shape_val[0] if hasattr(shape_val, "__len__") else shape_val

        # If genome length is still unset, infer from fitness spec (e.g. "sphere:dim=5").
        if genome_length is None and isinstance(fitness, str):
            from .catalog import OperatorCatalog

            try:
                _, parsed_params = OperatorCatalog().parse_spec(fitness)
                dim_val = parsed_params.get("dim", parsed_params.get("num_dims"))
                if dim_val is not None:
                    genome_length = int(dim_val)
            except Exception:
                pass

        if genome_type is None:
            genome_type = "real"
        if genome_length is None:
            genome_length = 10
        if bounds is None:
            bounds = (-5.0, 5.0)
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            bounds = (float(bounds[0]), float(bounds[1]))
        else:
            bounds = (-5.0, 5.0)

        pop_size = int(kwargs.get("pop_size", 50))
        population = PopulationConfig(
            size=pop_size,
            genome_type=str(genome_type),
            genome_length=int(genome_length),
            bounds=bounds,
            genome_spec=str(genome_spec) if genome_spec is not None else None,
        )

        # Execution config
        generations = int(kwargs.get("generations", 100))
        from .config import normalize_seeds

        raw_seeds = kwargs.get("seeds", (1, 2, 3))
        seeds = normalize_seeds(raw_seeds)
        maximize = bool(kwargs.get("maximize", False))
        prng_impl = kwargs.get("prng_impl")
        use_history_for_final = bool(kwargs.get("use_history_for_final", False))

        execution = ExecutionConfig(
            generations=generations,
            seeds=seeds,
            maximize=maximize,
            prng_impl=str(prng_impl) if prng_impl is not None else None,
            use_history_for_final=use_history_for_final,
        )

        # Logging config
        log_level = kwargs.get("log_level")
        log_interval = kwargs.get("log_interval")
        log_nan_watchdog = bool(kwargs.get("log_nan_watchdog", True))
        raw_metrics = kwargs.get("history_metrics")
        history_metrics = tuple(str(m) for m in raw_metrics) if raw_metrics is not None else None

        logging = LoggingConfig(
            log_level=str(log_level) if log_level is not None else None,
            log_interval=int(log_interval) if log_interval is not None else None,
            log_nan_watchdog=log_nan_watchdog,
            history_metrics=history_metrics,
        )

        # Output config
        experiment_name = str(kwargs.get("experiment_name", "quick_experiment"))
        raw_output_dir = kwargs.get("output_dir")
        output_dir = str(raw_output_dir) if raw_output_dir is not None else None
        raw_trace_dir = kwargs.get("trace_dir")
        trace_dir = str(raw_trace_dir) if raw_trace_dir is not None else None

        output = OutputConfig(
            experiment_name=experiment_name,
            output_dir=output_dir,
            trace_dir=trace_dir,
        )

        # Backend config
        raw_backend = kwargs.get("backend", "malthusjax")
        backend_name = str(raw_backend).lower()

        # Check if custom strategy routes to a different backend
        strategy_obj = kwargs.get("strategy")
        if strategy_obj is not None and backend_name == "malthusjax":
            from .backend_registry import get_backends

            _backends = get_backends()
            for name, entry in _backends.items():
                p = entry[0]
                if (
                    name != "malthusjax"
                    and hasattr(p, "handles_strategy")
                    and p.handles_strategy(strategy_obj)
                ):
                    backend_name = name
                    break

        # Known fields across top-level
        known_top_level = {
            "pop_size",
            "genome_type",
            "genome_length",
            "bounds",
            "genome",
            "generations",
            "seeds",
            "maximize",
            "prng_impl",
            "use_history_for_final",
            "log_level",
            "log_interval",
            "log_nan_watchdog",
            "history_metrics",
            "step_logging",
            "experiment_name",
            "output_dir",
            "trace_dir",
            "backend",
            "serialize_history",
            "engine",
            "seed",
        }

        backend: BackendConfig
        if backend_name in ("evosax", "composable_evosax"):
            strategy = kwargs.get("evosax_strategy", kwargs.get("strategy", "SimpleGA"))
            composable = (backend_name == "composable_evosax") or bool(
                kwargs.get("composable", False)
            )
            algo_kwargs = dict(kwargs.get("algorithm_kwargs", {}))
            known_evosax = known_top_level | {
                "evosax_strategy",
                "strategy",
                "composable",
                "algorithm_kwargs",
                "fitness",
            }
            extra = {k: v for k, v in kwargs.items() if k not in known_evosax}
            if strategy_obj is not None:
                extra["strategy"] = strategy_obj
            if "initial_population" in kwargs and kwargs["initial_population"] is not None:
                extra["initial_population"] = kwargs["initial_population"]
            backend = EvosaxBackendConfig(
                name="evosax",
                strategy=strategy,
                fitness=kwargs.get("fitness"),
                composable=composable,
                algorithm_kwargs=algo_kwargs,
                extra_kwargs=extra,
            )
        elif backend_name == "qdax":
            strategy_cls = kwargs.get("qdax_strategy", kwargs.get("strategy_cls", "MAPElites"))
            num_desc = int(kwargs.get("qdax_num_descriptors", kwargs.get("num_descriptors", 2)))
            num_cent = int(kwargs.get("qdax_num_centroids", kwargs.get("num_centroids", 100)))
            mut_sig = float(kwargs.get("qdax_mutation_sigma", kwargs.get("mutation_sigma", 0.1)))
            algo_kwargs = dict(kwargs.get("algorithm_kwargs", {}))
            known_qdax = known_top_level | {
                "qdax_strategy",
                "strategy_cls",
                "qdax_num_descriptors",
                "num_descriptors",
                "qdax_num_centroids",
                "num_centroids",
                "qdax_mutation_sigma",
                "mutation_sigma",
                "algorithm_kwargs",
                "fitness",
            }
            extra = {k: v for k, v in kwargs.items() if k not in known_qdax}
            if strategy_obj is not None:
                extra["strategy"] = strategy_obj
            if "initial_population" in kwargs and kwargs["initial_population"] is not None:
                extra["initial_population"] = kwargs["initial_population"]
            backend = QdaxBackendConfig(
                name="qdax",
                strategy_cls=strategy_cls,
                fitness=kwargs.get("fitness"),
                num_descriptors=num_desc,
                num_centroids=num_cent,
                mutation_sigma=mut_sig,
                algorithm_kwargs=algo_kwargs,
                extra_kwargs=extra,
            )
        elif backend_name in ("tensorneat", "composable_tensorneat"):
            algorithm = kwargs.get("tensorneat_algorithm", kwargs.get("algorithm", "NEAT"))
            genome_name = kwargs.get(
                "tensorneat_genome", kwargs.get("genome_name", "DefaultGenome")
            )
            problem = kwargs.get("tensorneat_problem", kwargs.get("problem"))
            num_in = int(kwargs.get("tensorneat_num_inputs", kwargs.get("num_inputs", 2)))
            num_out = int(kwargs.get("tensorneat_num_outputs", kwargs.get("num_outputs", 1)))
            composable = (backend_name == "composable_tensorneat") or bool(
                kwargs.get("composable", False)
            )
            algo_kwargs = dict(kwargs.get("algorithm_kwargs", {}))
            known_neat = known_top_level | {
                "tensorneat_algorithm",
                "algorithm",
                "tensorneat_genome",
                "genome_name",
                "tensorneat_problem",
                "problem",
                "tensorneat_num_inputs",
                "num_inputs",
                "tensorneat_num_outputs",
                "num_outputs",
                "composable",
                "algorithm_kwargs",
            }
            extra = {k: v for k, v in kwargs.items() if k not in known_neat}
            if strategy_obj is not None:
                extra["strategy"] = strategy_obj
            if "initial_population" in kwargs and kwargs["initial_population"] is not None:
                extra["initial_population"] = kwargs["initial_population"]
            backend = TensorneatBackendConfig(
                name="tensorneat",
                algorithm=algorithm,
                genome_name=genome_name,
                problem=problem,
                num_inputs=num_in,
                num_outputs=num_out,
                composable=composable,
                algorithm_kwargs=algo_kwargs,
                extra_kwargs=extra,
            )
        elif backend_name == "stub":
            known_stub = known_top_level
            extra = {k: v for k, v in kwargs.items() if k not in known_stub}
            backend = StubBackendConfig(name="stub", extra_kwargs=extra)
        elif backend_name == "malthusjax":
            # Default native malthusjax
            fitness = kwargs.get("fitness")
            selection = kwargs.get("selection")
            crossover = kwargs.get("crossover")
            mutation = kwargs.get("mutation")
            engine_type = kwargs.get("engine_type", "ga")
            elitism = int(kwargs.get("elitism", 2))
            data_config = kwargs.get("data_config")
            known_mjax = known_top_level | {
                "fitness",
                "selection",
                "crossover",
                "mutation",
                "engine_type",
                "elitism",
                "data_config",
                "strategy",
            }
            extra = {k: v for k, v in kwargs.items() if k not in known_mjax}
            if strategy_obj is not None:
                extra["strategy"] = strategy_obj
            if "initial_population" in kwargs and kwargs["initial_population"] is not None:
                extra["initial_population"] = kwargs["initial_population"]
            backend = MalthusJAXBackendConfig(
                name="malthusjax",
                fitness=fitness,
                selection=selection,
                crossover=crossover,
                mutation=mutation,
                engine_type=engine_type,
                elitism=elitism,
                genome=genome_spec,
                data_config=data_config,
                extra_kwargs=extra,
            )
        else:
            # Custom, plugin, or map_elites backend
            extra = {k: v for k, v in kwargs.items() if k not in known_top_level}
            if strategy_obj is not None:
                extra["strategy"] = strategy_obj
            if "initial_population" in kwargs and kwargs["initial_population"] is not None:
                extra["initial_population"] = kwargs["initial_population"]
            backend = GenericBackendConfig(
                name=raw_backend if isinstance(raw_backend, str) else backend_name,
                extra_kwargs=extra,
            )

        return cls(
            population=population,
            execution=execution,
            backend=backend,
            logging=logging,
            output=output,
        )

    @classmethod
    def from_toml(
        cls, path: str, pipelines: Optional[List[str]] = None
    ) -> Dict[str, "ExperimentConfig"]:
        """Load a TOML file and return a mapping of pipeline_name -> typed ExperimentConfig."""
        from .config import load_experiment_config

        res = load_experiment_config(path, pipelines=pipelines)
        meta = res.meta
        pipeline_defs = res.pipelines

        configs: Dict[str, ExperimentConfig] = {}
        for p_name, p_kwargs in pipeline_defs.items():
            merged = {**meta.get("shared", {}), **p_kwargs}
            if "experiment_name" not in merged and "name" in meta:
                merged["experiment_name"] = meta["name"]
            if "output_dir" not in merged and "output_dir" in meta:
                merged["output_dir"] = meta["output_dir"]
            configs[p_name] = cls.from_quick_run_kwargs(**merged)
        return configs

    def validate(self) -> None:
        """Validate config consistency.

        Raises
        ------
        ValueError
            If configuration contains invalid bounds, empty seeds,
            non-positive population size/generations, or operator/genome mismatches.
        """
        if self.population.bounds[0] >= self.population.bounds[1]:
            raise ValueError(
                f"Invalid bounds: lower ({self.population.bounds[0]}) must be strictly "
                f"less than upper ({self.population.bounds[1]})."
            )

        if not self.execution.seeds:
            raise ValueError("Execution seeds tuple must not be empty.")

        if self.population.size <= 0:
            raise ValueError(f"Population size must be positive, got {self.population.size}.")

        if self.execution.generations <= 0:
            raise ValueError(f"Generations must be positive, got {self.execution.generations}.")

        # Check operator compatibility for MalthusJAX backend
        if isinstance(self.backend, MalthusJAXBackendConfig):
            real_only_crossovers = ("blend", "simulated_binary", "uniform_real", "binomial")
            binary_only_crossovers = ("uniform_binary", "single_point")

            if self.backend.crossover:
                c_name = self.backend.crossover.split(":")[0].strip().lower()
                if self.population.genome_type == "binary" and any(
                    r in c_name for r in real_only_crossovers
                ):
                    raise ValueError(
                        f"Real-only crossover '{self.backend.crossover}' cannot be used with "
                        f"genome_type='{self.population.genome_type}'."
                    )
                if self.population.genome_type == "real" and any(
                    b in c_name for b in binary_only_crossovers
                ):
                    raise ValueError(
                        f"Binary-only crossover '{self.backend.crossover}' cannot be used with "
                        f"genome_type='{self.population.genome_type}'."
                    )
