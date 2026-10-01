"""Base classes and utilities for Scikit-Learn style estimators in malthus_gp.

Leverages MalthusJAX Composer and Engine catalogs for declarative execution,
multi-engine switching, and bidirectional TOML configuration serialization.
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
    from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin

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

    class TransformerMixin:  # type: ignore[no-redef]
        """Lightweight fallback for TransformerMixin."""

        def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
            return self.fit(X, y, **fit_params).transform(X)  # type: ignore[attr-defined]


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


class BaseGPEstimator(SklearnBaseEstimator):  # type: ignore[misc]
    """Base class for Genetic Programming estimators in malthus_gp.

    Provides Composer-driven dynamic spec generation, multi-engine execution,
    and bidirectional TOML configuration serialization.
    """

    # Registered estimator class mapping for from_toml deserialization
    _REGISTRY: ClassVar[Dict[str, Type["BaseGPEstimator"]]] = {}

    interpreter_: Any
    best_genome_: Any
    best_fitness_: float
    n_features_in_: int
    n_outputs_: int
    feature_names_in_: Optional[Sequence[str]]
    composed_specs_: Dict[str, str]
    config_: Any
    repertoire_: Any

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
            Number of target dimensions or synthesized features.

        Returns
        -------
        specs : dict of str -> str
            Dictionary containing 'genome', 'selection', 'crossover', 'mutation', and 'engine' specs.
        """
        is_diff = (
            getattr(self, "_is_differentiable", False)
            or getattr(self, "genome_type", None) == "diff_cartesian_genome"
            or hasattr(self, "lsmf_epochs")
        )

        n_rows = getattr(self, "n_rows", 1)
        n_cols = getattr(self, "n_cols", 40)
        levels_back = getattr(self, "levels_back", -1)
        pop_size = getattr(self, "pop_size", 5)
        max_generations = getattr(self, "max_generations", 2000)
        mutation_rate = getattr(self, "mutation_rate", None)
        if mutation_rate is None:
            mut_rate_val = 3.0 / (n_cols * 3)
        else:
            mut_rate_val = float(mutation_rate)

        # 1. Genome spec
        genome_prefix = "diff_cartesian_genome" if is_diff else "cartesian_genome"
        genome_spec = (
            f"{genome_prefix}:num_rows={n_rows},num_cols={n_cols},"
            f"num_inputs={n_inputs},num_outputs={n_outputs},levels_back={levels_back}"
        )

        # 2. Selection spec
        lambda_size = max(1, pop_size - 1)
        selection_spec = f"cgp_neutral:num_selections={lambda_size * 2},elite_k=1,n_elites=1"

        # 3. Crossover spec
        crossover_spec = "cartesian_noop"

        # 4. Mutation spec
        mutation_prefix = "diff_cartesian_mutation" if is_diff else "cartesian_micro"
        mutation_spec = f"{mutation_prefix}:mutation_rate={mut_rate_val:.6g}"

        # 5. Engine spec
        engine_str = getattr(self, "engine", "cgp")
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
        if engine_type == "island":
            if "num_islands" not in all_params:
                all_params["num_islands"] = 4
            if "migration_interval" not in all_params:
                all_params["migration_interval"] = 20
            if "num_migrants" not in all_params:
                all_params["num_migrants"] = 1
        elif engine_type == "map_elites":
            if "num_centroids" not in all_params:
                all_params["num_centroids"] = 50
        elif engine_type == "island_map_elites":
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
    ) -> "BaseGPEstimator":
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
            Evaluator class to instantiate. Defaults to CartesianGPEvaluator.
        evaluator_config_cls : class, optional
            Evaluator config class. Defaults to CartesianGPEvaluatorConfig.
        **evaluator_kwargs : Any
            Additional kwargs forwarded to evaluator.

        Returns
        -------
        self : BaseGPEstimator
            Fitted estimator instance.
        """
        from malthusjax.composer.catalog import OperatorCatalog
        from malthusjax.composer.engine_catalog import EngineRegistry
        from malthusjax.composer.genome_catalog import GenomeCatalog

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
        if isinstance(self, TransformerMixin) or hasattr(self, "n_features"):
            n_outputs = getattr(self, "n_features", 5)
            y_arr = np.asarray(y, dtype=np.float32).flatten() if y is not None else None
        elif y is not None:
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
        self.config_ = GenomeCatalog().get(specs["genome"])

        # 3. Ensure Interpreter is initialized
        is_diff = (
            getattr(self, "_is_differentiable", False)
            or getattr(self, "genome_type", None) == "diff_cartesian_genome"
            or hasattr(self, "lsmf_epochs")
        )

        if not hasattr(self, "interpreter_") or self.interpreter_ is None:
            if is_diff:
                from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
                    DifferentiableCartesianInterpreter,
                )

                interp_diff: Any = DifferentiableCartesianInterpreter
                diff_kwargs: Dict[str, Any] = {"num_inputs": n_features, "num_outputs": n_outputs}
                if hasattr(self, "activations") and getattr(self, "activations") is not None:
                    diff_kwargs["activations"] = getattr(self, "activations")
                if (
                    hasattr(self, "activation_names")
                    and getattr(self, "activation_names") is not None
                ):
                    diff_kwargs["activation_names"] = getattr(self, "activation_names")
                self.interpreter_ = interp_diff(**diff_kwargs)
            else:
                from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter

                interp_cgp: Any = CartesianInterpreter
                cgp_kwargs: Dict[str, Any] = {"num_inputs": n_features, "num_outputs": n_outputs}
                if hasattr(self, "functions") and getattr(self, "functions") is not None:
                    cgp_kwargs["functions"] = getattr(self, "functions")
                if hasattr(self, "function_names") and getattr(self, "function_names") is not None:
                    cgp_kwargs["function_names"] = getattr(self, "function_names")
                self.interpreter_ = interp_cgp(**cgp_kwargs)

        # 4. Resolve Evaluator
        if evaluator is None:
            if specs["engine"].startswith("map_elites") or specs["engine"].startswith(
                "island_map_elites"
            ):
                from malthus_gp.evaluators.cartesian_qd_evaluator import (
                    CartesianQDEvalConfig,
                    CartesianQDEvaluator,
                )

                evaluator_cls = CartesianQDEvaluator
                evaluator_config_cls = CartesianQDEvalConfig
            elif evaluator_cls is None:
                from malthus_gp.evaluators.cartesian_gp_evaluator import (
                    CartesianGPEvaluator,
                    CartesianGPEvaluatorConfig,
                )

                evaluator_cls = CartesianGPEvaluator
                evaluator_config_cls = CartesianGPEvaluatorConfig

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
                if "lsmf_epochs" in cfg_fields and hasattr(self, "lsmf_epochs"):
                    cfg_kwargs["lsmf_epochs"] = self.lsmf_epochs
                if "learning_rate" in cfg_fields and hasattr(self, "learning_rate"):
                    cfg_kwargs["learning_rate"] = self.learning_rate
                if "redundancy_weight" in cfg_fields and hasattr(self, "redundancy_weight"):
                    cfg_kwargs["redundancy_weight"] = self.redundancy_weight

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

        # 5. Resolve Operators via Catalog
        cat = OperatorCatalog()
        sel_op = cat.get(specs["selection"])
        cx_op = cat.get(specs["crossover"])
        mut_op = cat.get(specs["mutation"])

        # 6. Resolve Engine via EngineRegistry
        from malthusjax.engine.schedules import TrackBest

        engine_reg = EngineRegistry()
        genome_type_str = "diff_cartesian_genome" if is_diff else "cartesian_genome"
        engine_kwargs: Dict[str, Any] = {
            "evaluator": evaluator,
            "selection": sel_op,
            "crossover": cx_op,
            "mutation": mut_op,
            "genome_config": self.config_,
            "genome_type": genome_type_str,
            "pop_size": getattr(self, "pop_size", 5),
            "generations": getattr(self, "max_generations", 2000),
            "elitism": 1,
            "track_best": TrackBest.FULL,
        }
        if specs["engine"].startswith("map_elites") or specs["engine"].startswith(
            "island_map_elites"
        ):
            from malthus_gp.operators.cartesian_emitter import CartesianEmitter

            engine_kwargs["emitter"] = CartesianEmitter(
                mutation=mut_op,
                genome_config=self.config_,
                _batch_size=getattr(self, "pop_size", 5),
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

    def to_config_dict(self) -> Dict[str, Any]:
        """Exports the estimator configuration, hyperparameters, specs, and status to a dictionary."""
        params = self.get_params(deep=False)
        # Filter out unpicklable/unserializable callables (e.g. tuples of functions)
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
        if hasattr(self, "to_formula") and hasattr(self, "best_genome_"):
            try:
                model_dict["formula"] = self.to_formula()
            except Exception:
                pass

        if hasattr(self, "best_genome_") and self.best_genome_ is not None:
            bg = self.best_genome_
            genome_dict: Dict[str, Any] = {
                "ops": np.asarray(bg.ops).tolist(),
                "args": np.asarray(bg.args).tolist(),
                "out_nodes": np.asarray(bg.out_nodes).tolist(),
            }
            if hasattr(bg, "weights") and hasattr(bg, "biases"):
                genome_dict["weights"] = np.asarray(bg.weights).tolist()
                genome_dict["biases"] = np.asarray(bg.biases).tolist()
            model_dict["genome"] = genome_dict

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
    def from_toml(cls, toml_str_or_path: Union[str, Path]) -> "BaseGPEstimator":
        """Instantiates an estimator from a TOML configuration string or file path.

        Parameters
        ----------
        toml_str_or_path : str or Path
            TOML string content or path to an existing .toml file.

        Returns
        -------
        estimator : BaseGPEstimator
            Configured estimator instance.
        """
        if _tomllib is None:
            raise ImportError(
                "Cannot load TOML: Neither 'tomllib' nor 'tomli' is available in Python environment."
            )

        p = Path(toml_str_or_path) if isinstance(toml_str_or_path, (str, Path)) else None
        if p is not None and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(toml_str_or_path)

        data = _tomllib.loads(content)
        estimator_info = data.get("estimator", {})
        cls_name = estimator_info.get("class")

        target_cls: Type[BaseGPEstimator]
        if cls is not BaseGPEstimator:
            target_cls = cls
        elif cls_name and cls_name in BaseGPEstimator._REGISTRY:
            target_cls = BaseGPEstimator._REGISTRY[cls_name]
        else:
            raise ValueError(
                f"Unknown estimator class '{cls_name}' in TOML. Available: {list(BaseGPEstimator._REGISTRY.keys())}"
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
            g_data = model_info["genome"]
            n_in = getattr(inst, "n_features_in_", 1)
            n_out = getattr(inst, "n_outputs_", 1)

            if "weights" in g_data and "biases" in g_data:
                from malthus_gp.genomes.differentiable_cartesian_genome import (
                    DifferentiableCartesianGenome,
                )
                from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
                    DifferentiableCartesianInterpreter,
                )

                inst.best_genome_ = DifferentiableCartesianGenome(
                    ops=jnp.array(g_data["ops"], dtype=jnp.int32),
                    args=jnp.array(g_data["args"], dtype=jnp.int32),
                    out_nodes=jnp.array(g_data["out_nodes"], dtype=jnp.int32),
                    weights=jnp.array(g_data["weights"], dtype=jnp.float32),
                    biases=jnp.array(g_data["biases"], dtype=jnp.float32),
                )
                interp_diff: Any = DifferentiableCartesianInterpreter
                diff_kwargs: Dict[str, Any] = {"num_inputs": n_in, "num_outputs": n_out}
                if hasattr(inst, "activations") and getattr(inst, "activations") is not None:
                    diff_kwargs["activations"] = getattr(inst, "activations")
                if (
                    hasattr(inst, "activation_names")
                    and getattr(inst, "activation_names") is not None
                ):
                    diff_kwargs["activation_names"] = getattr(inst, "activation_names")
                inst.interpreter_ = interp_diff(**diff_kwargs)
            else:
                from malthus_gp.genomes.cartesian_genome import CartesianGenome
                from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter

                inst.best_genome_ = CartesianGenome(
                    ops=jnp.array(g_data["ops"], dtype=jnp.int32),
                    args=jnp.array(g_data["args"], dtype=jnp.int32),
                    out_nodes=jnp.array(g_data["out_nodes"], dtype=jnp.int32),
                )
                interp_cgp: Any = CartesianInterpreter
                cgp_kwargs: Dict[str, Any] = {"num_inputs": n_in, "num_outputs": n_out}
                if hasattr(inst, "functions") and getattr(inst, "functions") is not None:
                    cgp_kwargs["functions"] = getattr(inst, "functions")
                if hasattr(inst, "function_names") and getattr(inst, "function_names") is not None:
                    cgp_kwargs["function_names"] = getattr(inst, "function_names")
                inst.interpreter_ = interp_cgp(**cgp_kwargs)

        return inst


__all__ = [
    "BaseGPEstimator",
    "RegressorMixin",
    "ClassifierMixin",
    "TransformerMixin",
    "SKLEARN_AVAILABLE",
]
