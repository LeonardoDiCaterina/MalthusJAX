"""Command-line interface for Malthus-NEAT (mneat).

Provides commands to train neural network topologies, make predictions, visualize
evolved architectures, and export declarative TOML configurations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from malthus_neat.learn.base import BaseNEATEstimator
from malthus_neat.learn.neat_classifier import NEATClassifier
from malthus_neat.learn.neat_regressor import NEATRegressor
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.cli")


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
            y = (np.sin(X[:, 0] * 2.0) + 0.5 * X[:, 0]).astype(np.float32)
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
    """Executes the `mneat fit` command."""
    is_cls = args.estimator == "classifier"
    X, y, _ = _load_or_generate_data(args.data, target=args.target, is_classification=is_cls)

    engine_params = _parse_key_value_params(args.engine_params)

    estimator_kwargs: Dict[str, Any] = {
        "pop_size": args.pop_size,
        "max_generations": args.generations,
        "max_nodes": args.max_nodes,
        "max_conns": args.max_conns,
        "recurrent": args.recurrent,
        "elitism": args.elitism,
        "random_state": args.seed,
        "engine": args.engine,
        "engine_params": engine_params if engine_params else None,
    }

    if args.estimator == "classifier":
        model: BaseNEATEstimator = NEATClassifier(**estimator_kwargs)
    else:
        model = NEATRegressor(**estimator_kwargs)

    print(
        f"Fitting NEAT {args.estimator.upper()} using engine '{args.engine}' on {X.shape[0]} samples, {X.shape[1]} features..."
    )
    model.fit(X, y)

    print("Training completed!")
    print(f"Best fitness: {model.best_fitness_:.6f}")

    if args.output:
        out_p = Path(args.output)
        model.to_toml(out_p)
        print(f"Model and configuration saved to {out_p}")

    if args.predict_out:
        p_out = Path(args.predict_out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        preds = model.predict(X)
        if p_out.suffix == ".npy":
            np.save(p_out, preds)
        else:
            np.savetxt(p_out, preds, delimiter=",")
        print(f"Predictions saved to {p_out}")

    if args.visualize_out:
        v_out = Path(args.visualize_out)
        v_out.parent.mkdir(parents=True, exist_ok=True)
        model.visualize(filepath=v_out)
        print(f"Topology plot saved to {v_out}")

    return 0


def handle_predict(args: argparse.Namespace) -> int:
    """Executes the `mneat predict` command."""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model TOML file not found: {args.model}")

    model = BaseNEATEstimator.from_toml(model_path)
    X, _, _ = _load_or_generate_data(args.data, target=None)

    print(
        f"Loaded {model.__class__.__name__} from {model_path}. Generating predictions for {X.shape[0]} samples..."
    )
    preds = model.predict(X)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.suffix == ".npy":
        np.save(out_p, preds)
    else:
        np.savetxt(out_p, preds, delimiter=",")
    print(f"Predictions successfully saved to {out_p}")
    return 0


def handle_visualize(args: argparse.Namespace) -> int:
    """Executes the `mneat visualize` command."""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model TOML file not found: {args.model}")

    model = BaseNEATEstimator.from_toml(model_path)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "dot":
        dot_str = model.to_dot()
        out_p.write_text(dot_str, encoding="utf-8")
        print(f"DOT topology saved to {out_p}")
    else:
        model.visualize(filepath=out_p, format=args.format)
        print(f"Topology figure saved to {out_p}")
    return 0


def handle_export_toml(args: argparse.Namespace) -> int:
    """Executes the `mneat export-toml` command."""
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.estimator == "classifier":
        model: BaseNEATEstimator = NEATClassifier(engine=args.engine)
    else:
        model = NEATRegressor(engine=args.engine)

    model.composed_specs_ = model.compose_specs(n_inputs=2, n_outputs=1)
    model.to_toml(out_p)
    print(f"Template configuration exported to {out_p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the argument parser for the mneat CLI."""
    parser = argparse.ArgumentParser(
        prog="mneat",
        description="Malthus-NEAT: High-Performance Neuroevolution of Augmenting Topologies CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. FIT
    p_fit = subparsers.add_parser("fit", help="Train a NEAT model via Composer engines")
    p_fit.add_argument(
        "--estimator",
        choices=["regressor", "classifier"],
        default="regressor",
        help="Estimator type (default: regressor)",
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
        default="neat",
        help="Evolutionary engine: 'neat', 'island', 'map_elites', 'island_map_elites' (default: neat)",
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
        default=50,
        help="Maximum evolutionary generations (default: 50)",
    )
    p_fit.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=50,
        help="Population size (default: 50)",
    )
    p_fit.add_argument(
        "--max-nodes",
        type=int,
        default=20,
        help="Maximum node buffer capacity (default: 20)",
    )
    p_fit.add_argument(
        "--max-conns",
        type=int,
        default=40,
        help="Maximum connection buffer capacity (default: 40)",
    )
    p_fit.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Enable recurrent connections",
    )
    p_fit.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Number of elites preserved (default: 2)",
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
    p_fit.add_argument(
        "--visualize-out",
        type=str,
        default=None,
        help="Path to save network topology visualization",
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

    # 3. VISUALIZE
    p_vis = subparsers.add_parser("visualize", help="Render the topology diagram of a saved model")
    p_vis.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to trained model TOML",
    )
    p_vis.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save topology figure or DOT file",
    )
    p_vis.add_argument(
        "--format",
        choices=["svg", "png", "pdf", "dot"],
        default="svg",
        help="Output image format (default: svg)",
    )
    p_vis.set_defaults(func=handle_visualize)

    # 4. EXPORT-TOML
    p_exp = subparsers.add_parser("export-toml", help="Generate a template TOML configuration file")
    p_exp.add_argument(
        "--estimator",
        choices=["regressor", "classifier"],
        default="regressor",
        help="Estimator type (default: regressor)",
    )
    p_exp.add_argument(
        "--engine",
        type=str,
        default="neat",
        help="Engine type (default: neat)",
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
