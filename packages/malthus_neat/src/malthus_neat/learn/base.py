"""Base classes and utilities for Scikit-Learn style estimators in malthus_neat.

Leverages MalthusJAX Composer and Engine catalogs for declarative execution,
multi-engine switching (neat, island_neat, map_elites, island_map_elites),
and bidirectional TOML configuration serialization.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    import tomllib as _tomllib
except ModuleNotFoundError:
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore[assignment]

try:
    import toml as _toml_writer
except ModuleNotFoundError:
    _toml_writer = None  # type: ignore[assignment]

try:
    from sklearn.base import BaseEstimator as SklearnBaseEstimator
    from sklearn.base import ClassifierMixin, RegressorMixin

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    class SklearnBaseEstimator:  # type: ignore[no-redef]
        """Lightweight fallback for BaseEstimator when scikit-learn is not installed."""

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            init_signature = inspect.signature(self.__init__)  # type: ignore[misc]
            params: Dict[str, Any] = {}
            for name in init_signature.parameters:
                if name != "self":
                    params[name] = getattr(self, name, None)
            return params

        def set_params(self, **params: Any) -> "SklearnBaseEstimator":
            valid_params = self.get_params(deep=True)
            for key, value in params.items():
                if key in valid_params:
                    setattr(self, key, value)
            return self

    class RegressorMixin:  # type: ignore[no-redef]
        """Lightweight fallback for RegressorMixin."""

        _estimator_type = "regressor"

    class ClassifierMixin:  # type: ignore[no-redef]
        """Lightweight fallback for ClassifierMixin."""

        _estimator_type = "classifier"


def _dict_to_toml_str(d: Dict[str, Any]) -> str:
    """Zero-dependency fallback serializer from nested dictionary to valid TOML string."""
    lines: List[str] = []
    subsections: Dict[str, Dict[str, Any]] = {}

    for k, v in d.items():
        if isinstance(v, dict):
            subsections[k] = v
        elif isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, (list, tuple)):
            items = []
            for item in v:
                if isinstance(item, bool):
                    items.append("true" if item else "false")
                elif isinstance(item, (int, float)):
                    items.append(str(item))
                elif isinstance(item, str):
                    items.append(f'"{item}"')
            lines.append(f"{k} = [{', '.join(items)}]")

    for section_name, section_dict in subsections.items():
        lines.append(f"\n[{section_name}]")
        for k, v in section_dict.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, (list, tuple)):
                items = [f'"{i}"' if isinstance(i, str) else str(i) for i in v if not callable(i)]
                lines.append(f"{k} = [{', '.join(items)}]")

    return "\n".join(lines) + "\n"


class BaseNEATEstimator(SklearnBaseEstimator):  # type: ignore[misc]
    """Base class for NEAT neuroevolution estimators in malthus_neat.

    Provides Composer-driven dynamic spec generation, multi-engine execution
    (neat, island_neat, map_elites, island_map_elites), topology visualization,
    and bidirectional TOML configuration serialization.
    """

    # Registered estimator class mapping for from_toml deserialization
    _REGISTRY: ClassVar[Dict[str, Type["BaseNEATEstimator"]]] = {}

    interpreter_: Any
    best_genome_: Any
    best_fitness_: float
    n_features_in_: int
    n_outputs_: int
    feature_names_in_: Optional[Sequence[str]]
    composed_specs_: Dict[str, str]
    config_: Any
    repertoire_: Any
    history_: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._REGISTRY[cls.__name__] = cls

    def compose_specs(self, n_inputs: int, n_outputs: int = 1) -> Dict[str, str]:
        """Dynamically generates declarative MalthusJAX Composer DSL string specs.

        Parameters
        ----------
        n_inputs : int
            Number of input features.
        n_outputs : int, default=1
            Number of output targets.

        Returns
        -------
        specs : dict of str -> str
            Dictionary containing 'genome', 'selection', 'crossover', 'mutation', and 'engine' specs.
        """
        pop_size = getattr(self, "pop_size", 50)
        max_generations = getattr(self, "max_generations", 100)
        max_nodes = getattr(self, "max_nodes", 20)
        max_conns = getattr(self, "max_conns", 40)
        recurrent = getattr(self, "recurrent", False)
        elitism = getattr(self, "elitism", 2)

        # 1. Genome spec
        genome_spec = (
            f"neat_genome:num_inputs={n_inputs},num_outputs={n_outputs},"
            f"max_nodes={max_nodes},max_conns={max_conns},recurrent={str(recurrent).lower()}"
        )

        # 2. Selection spec
        selection_spec = f"tournament:tournament_size=3,num_selections={pop_size * 2}"

        # 3. Crossover spec
        crossover_spec = "neat_crossover"

        # 4. Mutation spec
        mutation_spec = "neat_mutation"

        # 5. Engine spec
        engine_str = getattr(self, "engine", "neat")
        engine_params_dict = dict(getattr(self, "engine_params", None) or {})

        if ":" in engine_str:
            base_engine, inline_params = engine_str.split(":", 1)
            for kv in inline_params.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k not in engine_params_dict:
                        try:
                            engine_params_dict[k] = int(v) if v.isdigit() else float(v)
                        except ValueError:
                            engine_params_dict[k] = v
            engine_type = base_engine
        else:
            engine_type = engine_str

        all_params = dict(engine_params_dict)
        if "pop_size" not in all_params:
            all_params["pop_size"] = pop_size
        if "generations" not in all_params:
            all_params["generations"] = max_generations
        if "elitism" not in all_params:
            all_params["elitism"] = elitism

        if engine_type in ("island", "island_neat"):
            engine_type = "island"
            all_params["genome_type"] = "neat_genome"
            if "num_islands" not in all_params:
                all_params["num_islands"] = 4
            if "migration_interval" not in all_params:
                all_params["migration_interval"] = 20
            if "num_migrants" not in all_params:
                all_params["num_migrants"] = 2
        elif engine_type == "map_elites":
            all_params["genome_type"] = "neat_genome"
            if "num_centroids" not in all_params:
                all_params["num_centroids"] = 50
        elif engine_type == "island_map_elites":
            all_params["genome_type"] = "neat_genome"
            if "num_islands" not in all_params:
                all_params["num_islands"] = 4
            if "migration_interval" not in all_params:
                all_params["migration_interval"] = 20
            if "num_migrants" not in all_params:
                all_params["num_migrants"] = 2
            if "num_centroids" not in all_params:
                all_params["num_centroids"] = 50

        param_str = ",".join(f"{k}={v}" for k, v in all_params.items())
        engine_spec = f"{engine_type}:{param_str}" if param_str else engine_type

        return {
            "genome": genome_spec,
            "selection": selection_spec,
            "crossover": crossover_spec,
            "mutation": mutation_spec,
            "engine": engine_spec,
        }

    def fit_via_composer(
        self,
        X: Any,
        y: Any = None,
        evaluator: Any = None,
        evaluator_cls: Any = None,
        evaluator_config_cls: Any = None,
        **evaluator_kwargs: Any,
    ) -> "BaseNEATEstimator":
        """Executes model training using MalthusJAX Composer string specifications and EngineRegistry.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, optional
            Target values.
        evaluator : Any, optional
            Pre-instantiated evaluator.
        evaluator_cls : class, optional
            Evaluator class to instantiate.
        evaluator_config_cls : class, optional
            Evaluator config class.
        **evaluator_kwargs : Any
            Additional kwargs forwarded to evaluator.

        Returns
        -------
        self : BaseNEATEstimator
            Fitted estimator instance.
        """
        from malthus_neat.genomes.neat_genome import NeatGenomeConfig
        from malthus_neat.interpreters.neat_interpreter import NeatInterpreter
        from malthus_neat.operators.neat_crossover import NeatCrossover
        from malthus_neat.operators.neat_mutation import NeatMutation
        from malthusjax.composer.catalog import OperatorCatalog
        from malthusjax.composer.engine_catalog import EngineRegistry

        # Convert X to numpy float32
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.feature_names_in_: Optional[Sequence[str]] = (
            list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(n_features)]
        )

        # Determine number of outputs
        y_arr: Any = None
        if y is not None:
            raw_y = np.asarray(y, dtype=np.float32)
            if raw_y.ndim == 1:
                y_arr = raw_y.reshape(-1, 1)
            else:
                y_arr = raw_y
            n_outputs = int(y_arr.shape[1])
        else:
            y_arr = None
            n_outputs = 1

        self.n_outputs_ = n_outputs

        # 1. Compose specs
        specs = self.compose_specs(n_inputs=n_features, n_outputs=n_outputs)
        self.composed_specs_ = specs

        # 2. Resolve Genome Config
        self.config_ = NeatGenomeConfig(
            num_inputs=n_features,
            num_outputs=n_outputs,
            max_nodes=getattr(self, "max_nodes", 20),
            max_conns=getattr(self, "max_conns", 40),
            recurrent=getattr(self, "recurrent", False),
        )

        # 3. Resolve Interpreter
        if not hasattr(self, "interpreter_") or self.interpreter_ is None:
            self.interpreter_ = NeatInterpreter(genome_config=self.config_)

        # 4. Resolve Evaluator
        is_qd = specs["engine"].startswith("map_elites") or specs["engine"].startswith(
            "island_map_elites"
        )
        if evaluator is None:
            if is_qd:
                from malthus_neat.evaluators.neat_qd_evaluator import (
                    NeatQDEvalConfig,
                    NeatQDEvaluator,
                )

                evaluator_cls = NeatQDEvaluator
                evaluator_config_cls = NeatQDEvalConfig
            elif evaluator_cls is None:
                from malthus_neat.evaluators.neat_evaluator import (
                    NeatEvaluator,
                    NeatEvaluatorConfig,
                )

                evaluator_cls = NeatEvaluator
                evaluator_config_cls = NeatEvaluatorConfig

            eval_data = (
                (jnp.array(X_arr), jnp.array(y_arr)) if y is not None else (jnp.array(X_arr),)
            )

            if evaluator_config_cls is not None:
                cfg_fields = {f.name for f in dataclasses.fields(evaluator_config_cls)}
                cfg_kwargs: Dict[str, Any] = {}
                if "genome_config" in cfg_fields:
                    cfg_kwargs["genome_config"] = self.config_
                if "maximize" in cfg_fields:
                    cfg_kwargs["maximize"] = getattr(self, "maximize", False)
                if "loss_function" in cfg_fields and hasattr(self, "loss_function"):
                    cfg_kwargs["loss_function"] = self.loss_function

                for k, v in evaluator_kwargs.items():
                    if k in cfg_fields:
                        cfg_kwargs[k] = v

                eval_config = evaluator_config_cls(**cfg_kwargs)
                evaluator = evaluator_cls(
                    config=eval_config,
                    data=eval_data,
                    interpreter=self.interpreter_,
                )
            else:
                evaluator = evaluator_cls(
                    data=eval_data,
                    interpreter=self.interpreter_,
                    **evaluator_kwargs,
                )

        # 5. Resolve Operators via Catalog and Configured Instances
        cat = OperatorCatalog()
        sel_op = cat.get(specs["selection"])
        cx_op = NeatCrossover(genome_config=self.config_)
        mut_op = NeatMutation(genome_config=self.config_)

        # 6. Resolve Engine via EngineRegistry
        from malthusjax.engine.schedules import TrackBest

        engine_reg = EngineRegistry()
        engine_kwargs: Dict[str, Any] = {
            "evaluator": evaluator,
            "selection": sel_op,
            "crossover": cx_op,
            "mutation": mut_op,
            "genome_config": self.config_,
            "genome_type": "neat_genome",
            "pop_size": getattr(self, "pop_size", 50),
            "generations": getattr(self, "max_generations", 100),
            "elitism": getattr(self, "elitism", 2),
            "track_best": TrackBest.FULL,
        }
        if specs["engine"].startswith("map_elites") or specs["engine"].startswith(
            "island_map_elites"
        ):
            from malthus_neat.operators.neat_emitter import NeatEmitter

            engine_kwargs["emitter"] = NeatEmitter(
                genome_config=self.config_,
                _batch_size=getattr(self, "pop_size", 50),
                mutation=mut_op,
            )
        engine = engine_reg.get(specs["engine"], **engine_kwargs)

        # 7. Execute Evolution
        key = jax.random.key(getattr(self, "random_state", 42))
        init_state = engine.init_state(key)
        final_state, output, _ = engine.run(init_state)

        self.best_genome_ = final_state.best_genome
        self.best_fitness_ = float(final_state.best_fitness)
        self.history_ = output
        if hasattr(final_state, "repertoire"):
            self.repertoire_ = final_state.repertoire

        return self

    def get_repertoire(self) -> Any:
        """Returns the Quality-Diversity MAP-Elites repertoire if trained with a QD engine."""
        if not hasattr(self, "repertoire_"):
            raise ValueError(
                "Estimator does not have a repertoire. Train with engine='map_elites' or engine='island_map_elites'."
            )
        return self.repertoire_

    def visualize(
        self,
        filepath: Optional[Union[str, Path]] = None,
        format: str = "svg",
        view: bool = False,
    ) -> Any:
        """Renders and optionally saves the evolved neural network topology.

        Parameters
        ----------
        filepath : str or Path, optional
            Path where the topology diagram should be saved.
        format : str, default="svg"
            Image format ('svg', 'png', 'pdf').
        view : bool, default=False
            Whether to show interactive matplotlib window.
        """
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'visualize'.")

        from malthus_neat.interpreters.visualizer import draw_network

        path_str = str(filepath) if filepath is not None else None
        return draw_network(
            self.best_genome_,
            config=self.config_,
            filepath=path_str,
            format=format,
            feature_names=getattr(self, "feature_names_in_", None),
            view=view,
        )

    def plot_network(
        self,
        path: Optional[str] = None,
        view: bool = False,
        ax: Optional[Any] = None,
    ) -> Any:
        """Renders the evolved neural network topology plot using matplotlib."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'plot_network'.")

        from malthus_neat.interpreters.visualizer import plot_network

        return plot_network(self.best_genome_, self.config_, path=path, view=view, ax=ax)

    def to_dot(self) -> str:
        """Generates Graphviz DOT script of the evolved network."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        from malthus_neat.interpreters.visualizer import to_dot

        return to_dot(self.best_genome_, self.config_)

    def to_networkx(self) -> Any:
        """Converts the evolved network topology to a NetworkX DiGraph."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        from malthus_neat.interpreters.visualizer import to_networkx

        return to_networkx(self.best_genome_, self.config_)

    def to_config_dict(self) -> Dict[str, Any]:
        """Exports the estimator configuration, hyperparameters, specs, and status to a dictionary."""
        params = self.get_params(deep=False)
        clean_params: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                clean_params[k] = v
            elif isinstance(v, (list, tuple)) and all(
                isinstance(i, (int, float, str, bool)) for i in v
            ):
                clean_params[k] = list(v)

        model_dict: Dict[str, Any] = {}
        if hasattr(self, "best_fitness_"):
            model_dict["best_fitness"] = float(self.best_fitness_)
        if hasattr(self, "n_features_in_"):
            model_dict["n_features_in"] = int(self.n_features_in_)
        if hasattr(self, "n_outputs_"):
            model_dict["n_outputs"] = int(self.n_outputs_)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            model_dict["feature_names_in"] = list(self.feature_names_in_)

        if hasattr(self, "best_genome_") and self.best_genome_ is not None:
            bg = self.best_genome_
            model_dict["genome"] = {
                "nodes": np.asarray(bg.nodes).tolist(),
                "conns": np.asarray(bg.conns).tolist(),
            }

        return {
            "estimator": {
                "class": self.__class__.__name__,
                "module": self.__class__.__module__,
            },
            "params": clean_params,
            "specs": getattr(self, "composed_specs_", {}),
            "model": model_dict,
        }

    def to_toml(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Serializes estimator configuration and specs to a TOML string, and optionally writes to file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to write TOML content to.

        Returns
        -------
        toml_content : str
            Formatted TOML text.
        """
        cfg = self.to_config_dict()
        if _toml_writer is not None:
            content = _toml_writer.dumps(cfg)
        else:
            content = _dict_to_toml_str(cfg)

        if filepath is not None:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        return content

    @classmethod
    def from_toml(cls, toml_str_or_path: Union[str, Path]) -> "BaseNEATEstimator":
        """Instantiates an estimator from a TOML configuration string or file path.

        Parameters
        ----------
        toml_str_or_path : str or Path
            TOML string content or path to an existing .toml file.

        Returns
        -------
        estimator : BaseNEATEstimator
            Configured estimator instance.
        """
        if _tomllib is None:
            raise ImportError(
                "Cannot load TOML: Neither 'tomllib' nor 'tomli' is available in Python environment."
            )

        toml_str = str(toml_str_or_path)
        if "\n" not in toml_str and len(toml_str) < 1024:
            try:
                p = Path(toml_str)
                if p.is_file():
                    content = p.read_text(encoding="utf-8")
                else:
                    content = toml_str
            except (OSError, ValueError):
                content = toml_str
        else:
            content = toml_str

        data = _tomllib.loads(content)
        estimator_info = data.get("estimator", {})
        cls_name = estimator_info.get("class")

        target_cls: Type[BaseNEATEstimator]
        if cls is not BaseNEATEstimator:
            target_cls = cls
        elif cls_name and cls_name in BaseNEATEstimator._REGISTRY:
            target_cls = BaseNEATEstimator._REGISTRY[cls_name]
        else:
            raise ValueError(
                f"Unknown estimator class '{cls_name}' in TOML. Available: {list(BaseNEATEstimator._REGISTRY.keys())}"
            )

        params = data.get("params", {})
        inst = target_cls(**params)
        if "specs" in data:
            inst.composed_specs_ = data["specs"]

        model_info = data.get("model", {})
        if "best_fitness" in model_info:
            inst.best_fitness_ = float(model_info["best_fitness"])
        if "n_features_in" in model_info:
            inst.n_features_in_ = int(model_info["n_features_in"])
        if "n_outputs" in model_info:
            inst.n_outputs_ = int(model_info["n_outputs"])
        if "feature_names_in" in model_info and model_info["feature_names_in"] is not None:
            inst.feature_names_in_ = list(model_info["feature_names_in"])
        else:
            inst.feature_names_in_ = None

        if "genome" in model_info:
            from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
            from malthus_neat.interpreters.neat_interpreter import NeatInterpreter

            g_data = model_info["genome"]
            inst.best_genome_ = NeatGenome(
                nodes=jnp.array(g_data["nodes"], dtype=jnp.float32),
                conns=jnp.array(g_data["conns"], dtype=jnp.float32),
            )
            inst.config_ = NeatGenomeConfig(
                num_inputs=getattr(inst, "n_features_in_", 2),
                num_outputs=getattr(inst, "n_outputs_", 1),
                max_nodes=getattr(inst, "max_nodes", 20),
                max_conns=getattr(inst, "max_conns", 40),
                recurrent=getattr(inst, "recurrent", False),
            )
            inst.interpreter_ = NeatInterpreter(genome_config=inst.config_)

        return inst


__all__ = [
    "BaseNEATEstimator",
    "RegressorMixin",
    "ClassifierMixin",
    "SKLEARN_AVAILABLE",
]
