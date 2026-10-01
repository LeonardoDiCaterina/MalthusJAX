"""04_sklearn_pipeline_synthesis.py.

Demonstrates automated nonlinear feature synthesis using SymbolicFeatureTransformer
within a standard Scikit-Learn Pipeline, comparing linear baseline vs. augmented features.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_gp.learn import SymbolicFeatureTransformer


def run_pipeline_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-GP Showcase 4: Scikit-Learn Pipeline Feature Synthesis")
    print("=" * 70)

    # Generate non-linear synthetic regression task with interaction terms
    np.random.seed(42)
    n_samples = 250
    X = np.random.uniform(-2.0, 2.0, size=(n_samples, 3))
    # True relationship: y = 2.0 * x0 * x1 + sin(3 * x2) - 1.5 * x0^2
    y = (
        2.0 * X[:, 0] * X[:, 1]
        + np.sin(3.0 * X[:, 2])
        - 1.5 * (X[:, 0] ** 2)
        + np.random.normal(0, 0.1, size=n_samples)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Dataset: {n_samples} samples, 3 features [x0, x1, x2]")
    print("Target:  y = 2.0 * x0 * x1 + sin(3 * x2) - 1.5 * x0² (+ noise)")

    # 1. Baseline: Pure Linear Ridge Regression (Cannot model cross-terms without manual engineering)
    print("\n--- 1. Baseline: Standard Linear Ridge Regression ---")
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    r2_base_train = baseline.score(X_train, y_train)
    r2_base_test = baseline.score(X_test, y_test)
    print(f"Baseline Train R²: {r2_base_train:.4f}")
    print(f"Baseline Test R²:  {r2_base_test:.4f} (Poor fit due to non-linearity)")

    # 2. Automated Pipeline: SymbolicFeatureTransformer + Ridge Regression
    print("\n--- 2. Pipeline: SymbolicFeatureTransformer (5 features) + Ridge ---")
    t0 = time.time()
    pipe = Pipeline(
        [
            (
                "sym_features",
                SymbolicFeatureTransformer(
                    n_features=5,
                    include_original=True,
                    redundancy_weight=0.5,
                    max_generations=150,
                    pop_size=5,
                    random_state=42,
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)
    elapsed = time.time() - t0

    r2_pipe_train = pipe.score(X_train, y_train)
    r2_pipe_test = pipe.score(X_test, y_test)
    print(f"Feature Synthesis Time: {elapsed:.3f} s")
    print(f"Pipeline Train R²:      {r2_pipe_train:.4f}")
    print(f"Pipeline Test R²:       {r2_pipe_test:.4f} (Significant accuracy boost)")

    # Inspect synthesized mathematical expressions
    transformer = pipe.named_steps["sym_features"]
    print("\nSynthesized Non-Linear Mathematical Features:")
    for i, formula in enumerate(transformer.to_formula()):
        print(f"  Feature φ_{i}(X): {formula}")

    # Plot parity comparison
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    y_pred_base = baseline.predict(X_test)
    y_pred_pipe = pipe.predict(X_test)

    # Baseline Parity
    axes[0].scatter(y_test, y_pred_base, color="#d62728", alpha=0.7, edgecolors="none")
    min_val = min(float(np.min(y_test)), float(np.min(y_pred_base)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred_base)))
    axes[0].plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5)
    axes[0].set_title(f"Baseline Linear Ridge (R² = {r2_base_test:.3f})", fontsize=11)
    axes[0].set_xlabel("True Target y")
    axes[0].set_ylabel("Predicted y")
    axes[0].grid(True, alpha=0.3)

    # Pipeline Parity
    axes[1].scatter(y_test, y_pred_pipe, color="#1f77b4", alpha=0.7, edgecolors="none")
    min_val2 = min(float(np.min(y_test)), float(np.min(y_pred_pipe)))
    max_val2 = max(float(np.max(y_test)), float(np.max(y_pred_pipe)))
    axes[1].plot([min_val2, max_val2], [min_val2, max_val2], "k--", lw=1.5)
    axes[1].set_title(f"Symbolic Feature Pipeline + Ridge (R² = {r2_pipe_test:.3f})", fontsize=11)
    axes[1].set_xlabel("True Target y")
    axes[1].set_ylabel("Predicted y")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved pipeline parity plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "04_feature_synthesis.png"
    run_pipeline_showcase(plot_file)
