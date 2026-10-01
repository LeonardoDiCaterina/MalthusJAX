"""Command-line interface for Malthus-GP (mgp).

Provides commands to train symbolic models, make predictions, extract mathematical
formulas, and generate or inspect declarative TOML configurations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from malthus_gp.learn.base import BaseGPEstimator
from malthus_gp.learn.cgp_classifier import CGPClassifier
from malthus_gp.learn.cgp_regressor import CGPRegressor
from malthus_gp.learn.dcgp_regressor import DCGPRegressor
from malthus_gp.learn.symbolic_transformer import SymbolicFeatureTransformer
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.cli")


def _parse_key_value_params(params_str: Optional[str]) -> Dict[str, Any]:
    """Parses a comma-separated key=value string into a typed dictionary."""
    if not params_str:
        return {}
    params: Dict[str, Any] = {}
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid parameter format: '{part}'. Expected 'key=value'")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Parse value type
        if v.lower() == "true":
            params[k] = True
        elif v.lower() == "false":
            params[k] = False
        else:
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return params


def _load_or_generate_data(
    data_path: str, target: Any = -1, is_classification: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Loads feature matrix X and target y from file or generates synthetic benchmark data."""
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None

    if data_path == "synthetic" or not data_path:
        np.random.seed(42)
        X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
        feature_names = ["x0"]
        if is_classification:
            y = (X[:, 0] > 0.0).astype(int)
        else:
            y = (X[:, 0] ** 3 - 0.5 * X[:, 0] ** 2 + 1.2).astype(np.float32)
        return X, y, feature_names

    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if p.suffix in (".npy", ".npz"):
        loaded = np.load(p)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            X = loaded["X"] if "X" in loaded else loaded["arr_0"]
            y = loaded["y"] if "y" in loaded else None
        else:
            X = loaded
            y = None
        return X, y, None

    # CSV / TSV handling
    try:
        import pandas as pd

        df = pd.read_csv(p)
        feature_cols = list(df.columns)
        if isinstance(target, str) and target in df.columns:
            y = df[target].values
            feature_cols.remove(target)
            X = df[feature_cols].values
            feature_names = feature_cols
        elif isinstance(target, int):
            t_idx = target if target >= 0 else len(df.columns) + target
            y_col = df.columns[t_idx]
            y = df[y_col].values
            feature_cols.remove(y_col)
            X = df[feature_cols].values
            feature_names = feature_cols
        else:
            X = df.values
            y = None
            feature_names = list(df.columns)
    except ImportError:
        raw_arr = np.genfromtxt(p, delimiter=",", skip_header=1)
        if np.isnan(raw_arr).any():
            raw_arr = np.genfromtxt(p, delimiter=",")
        if target is not None:
            t_idx = int(target) if isinstance(target, int) else -1
            X = np.delete(raw_arr, t_idx, axis=1)
            y = raw_arr[:, t_idx]
        else:
            X = raw_arr
            y = None

    return X, y, feature_names


def handle_fit(args: argparse.Namespace) -> int:
    """Executes the `mgp fit` command."""
    is_cls = args.estimator == "classifier"
    X, y, feature_names = _load_or_generate_data(
        args.data, target=args.target, is_classification=is_cls
    )

    engine_params = _parse_key_value_params(args.engine_params)

    estimator_kwargs: Dict[str, Any] = {
        "n_rows": args.n_rows,
        "n_cols": args.n_cols,
        "levels_back": args.levels_back,
        "pop_size": args.pop_size,
        "max_generations": args.generations,
        "random_state": args.seed,
        "engine": args.engine,
        "engine_params": engine_params if engine_params else None,
    }

    if args.estimator == "cgp":
        model: BaseGPEstimator = CGPRegressor(**estimator_kwargs)
    elif args.estimator == "dcgp":
        estimator_kwargs["lsmf_epochs"] = args.lsmf_epochs
        estimator_kwargs["learning_rate"] = args.learning_rate
        model = DCGPRegressor(**estimator_kwargs)
    elif args.estimator == "classifier":
        model = CGPClassifier(**estimator_kwargs)
    elif args.estimator == "transformer":
        estimator_kwargs["n_features"] = args.n_features
        model = SymbolicFeatureTransformer(**estimator_kwargs)
    else:
        raise ValueError(f"Unknown estimator: '{args.estimator}'")

    print(
        f"Fitting {args.estimator.upper()} using engine '{args.engine}' on {X.shape[0]} samples, {X.shape[1]} features..."
    )
    model.fit(X, y)

    print("Training completed!")
    print(f"Best fitness: {model.best_fitness_:.6f}")

    if hasattr(model, "to_formula"):
        print("\nDecoded Formula:")
        f = model.to_formula(feature_names=feature_names)
        if isinstance(f, list):
            for i, expr in enumerate(f):
                print(f"  feat_{i} = {expr}")
        else:
            print(f"  y = {f}")

    if args.output:
        out_p = Path(args.output)
        model.to_toml(out_p)
        print(f"\nModel and configuration saved to {out_p}")

    if args.predict_out:
        p_out = Path(args.predict_out)
        preds = model.predict(X) if hasattr(model, "predict") else model.transform(X)
        if p_out.suffix == ".npy":
            np.save(p_out, preds)
        else:
            np.savetxt(p_out, preds, delimiter=",")
        print(f"Predictions saved to {p_out}")

    return 0


def handle_predict(args: argparse.Namespace) -> int:
    """Executes the `mgp predict` command."""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model TOML file not found: {args.model}")

    model = BaseGPEstimator.from_toml(model_path)
    X, _, _ = _load_or_generate_data(args.data, target=None)

    print(
        f"Loaded {model.__class__.__name__} from {model_path}. Generating predictions for {X.shape[0]} samples..."
    )
    preds: np.ndarray
    if hasattr(model, "predict"):
        preds = model.predict(X)
    elif hasattr(model, "transform"):
        preds = model.transform(X)
    else:
        raise ValueError(f"Loaded model of type {type(model)} has neither predict nor transform.")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.suffix == ".npy":
        np.save(out_p, preds)
    else:
        np.savetxt(out_p, preds, delimiter=",")
    print(f"Predictions successfully saved to {out_p}")
    return 0


def handle_formula(args: argparse.Namespace) -> int:
    """Executes the `mgp formula` command."""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model TOML file not found: {args.model}")

    model = BaseGPEstimator.from_toml(model_path)
    print(f"--- Model Formula ({model.__class__.__name__}) ---")
    if hasattr(model, "best_fitness_"):
        print(f"Best fitness: {model.best_fitness_:.6f}")

    if hasattr(model, "to_formula"):
        f = model.to_formula()
        if isinstance(f, list):
            for i, expr in enumerate(f):
                print(f"feat_{i} = {expr}")
        else:
            print(f"y = {f}")
    else:
        print("Model does not expose symbolic formula.")
    return 0


def handle_export_toml(args: argparse.Namespace) -> int:
    """Executes the `mgp export-toml` command."""
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.estimator == "cgp":
        model: BaseGPEstimator = CGPRegressor(engine=args.engine)
    elif args.estimator == "dcgp":
        model = DCGPRegressor(engine=args.engine)
    elif args.estimator == "classifier":
        model = CGPClassifier(engine=args.engine)
    elif args.estimator == "transformer":
        model = SymbolicFeatureTransformer(engine=args.engine)
    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")

    # Generate template specs for 2 inputs, 1 output
    model.composed_specs_ = model.compose_specs(n_inputs=2, n_outputs=1)
    model.to_toml(out_p)
    print(f"Template configuration exported to {out_p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the argument parser for the mgp CLI."""
    parser = argparse.ArgumentParser(
        prog="mgp",
        description="Malthus-GP: High-Performance Symbolic Genetic Programming CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. FIT
    p_fit = subparsers.add_parser("fit", help="Train a symbolic model via Composer engines")
    p_fit.add_argument(
        "--estimator",
        choices=["cgp", "dcgp", "classifier", "transformer"],
        default="cgp",
        help="Estimator type (default: cgp)",
    )
    p_fit.add_argument(
        "--data",
        "-d",
        type=str,
        default="synthetic",
        help="Path to CSV/NPY dataset or 'synthetic' (default: synthetic)",
    )
    p_fit.add_argument(
        "--target",
        type=str,
        default="-1",
        help="Target column name or index in dataset (default: -1)",
    )
    p_fit.add_argument(
        "--engine",
        type=str,
        default="cgp",
        help="Evolutionary engine: 'cgp', 'island', etc. (default: cgp)",
    )
    p_fit.add_argument(
        "--engine-params",
        type=str,
        default=None,
        help="Key=value engine parameters (e.g. 'num_islands=4,migration_interval=20')",
    )
    p_fit.add_argument(
        "--generations",
        "-g",
        type=int,
        default=200,
        help="Maximum evolutionary generations (default: 200)",
    )
    p_fit.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=5,
        help="Population size (1 + λ) (default: 5)",
    )
    p_fit.add_argument(
        "--n-cols",
        "-c",
        type=int,
        default=30,
        help="Number of columns in Cartesian DAG grid (default: 30)",
    )
    p_fit.add_argument(
        "--n-rows",
        "-r",
        type=int,
        default=1,
        help="Number of rows in Cartesian DAG grid (default: 1)",
    )
    p_fit.add_argument(
        "--levels-back",
        "-l",
        type=int,
        default=-1,
        help="Connection locality window (-1 for full connectivity) (default: -1)",
    )
    p_fit.add_argument(
        "--lsmf-epochs",
        type=int,
        default=10,
        help="Inner Adam gradient descent epochs for DCGP (default: 10)",
    )
    p_fit.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Continuous optimizer learning rate for DCGP (default: 0.02)",
    )
    p_fit.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Number of features to generate for transformer (default: 5)",
    )
    p_fit.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="PRNG seed (default: 42)",
    )
    p_fit.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save trained model TOML",
    )
    p_fit.add_argument(
        "--predict-out",
        type=str,
        default=None,
        help="Path to save model predictions on training data",
    )
    p_fit.set_defaults(func=handle_fit)

    # 2. PREDICT
    p_pred = subparsers.add_parser("predict", help="Generate predictions using a saved model TOML")
    p_pred.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to trained model TOML",
    )
    p_pred.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to test CSV/NPY dataset",
    )
    p_pred.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save generated predictions (CSV/NPY)",
    )
    p_pred.set_defaults(func=handle_predict)

    # 3. FORMULA
    p_form = subparsers.add_parser(
        "formula", help="Display the mathematical formula of a saved model"
    )
    p_form.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to trained model TOML",
    )
    p_form.set_defaults(func=handle_formula)

    # 4. EXPORT-TOML
    p_exp = subparsers.add_parser("export-toml", help="Generate a template TOML configuration file")
    p_exp.add_argument(
        "--estimator",
        choices=["cgp", "dcgp", "classifier", "transformer"],
        default="cgp",
        help="Estimator type (default: cgp)",
    )
    p_exp.add_argument(
        "--engine",
        type=str,
        default="cgp",
        help="Engine type (default: cgp)",
    )
    p_exp.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to write template TOML file",
    )
    p_exp.set_defaults(func=handle_export_toml)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        res = args.func(args)
        return int(res) if res is not None else 0
    except Exception as e:
        logger.error("CLI error: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
