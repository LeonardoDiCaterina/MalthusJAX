"""Scikit-Learn compatible Cartesian Genetic Programming Classifier."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from malthus_gp.core.functions import ARITHMETIC_4_FUNCTIONS, ARITHMETIC_4_NAMES
from malthus_gp.evaluators.cartesian_gp_evaluator import (
    CartesianGPEvaluator,
    CartesianGPEvaluatorConfig,
)
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthus_gp.learn.base import BaseGPEstimator, ClassifierMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.learn.cgp_classifier")


class CGPClassifier(BaseGPEstimator, ClassifierMixin):  # type: ignore[misc]
    """Cartesian Genetic Programming Classifier.

    Evolves feed-forward DAG logic graphs for classification using the (1 + λ)-ES
    strategy with neutral drift.

    Parameters
    ----------
    n_rows : int, default=1
        Number of rows in the Cartesian grid.
    n_cols : int, default=40
        Number of columns in the Cartesian grid.
    levels_back : int, default=-1
        Connection locality window (-1 for full connectivity).
    pop_size : int, default=5
        Total population size (1 + λ).
    max_generations : int, default=2000
        Maximum evolutionary generations.
    mutation_rate : float or None, default=None
        Per-gene point mutation rate. If None, targets ~3 mutated genes per chromosome.
    functions : tuple of callables, default=ARITHMETIC_4_FUNCTIONS
        Tuple of primitive functions.
    function_names : tuple of str, default=ARITHMETIC_4_NAMES
        Corresponding string symbols.
    random_state : int, default=42
        PRNG seed.
    """

    def __init__(
        self,
        n_rows: int = 1,
        n_cols: int = 40,
        levels_back: int = -1,
        pop_size: int = 5,
        max_generations: int = 2000,
        mutation_rate: Optional[float] = None,
        functions: Tuple[Callable[[jax.Array, jax.Array], jax.Array], ...] = ARITHMETIC_4_FUNCTIONS,
        function_names: Tuple[str, ...] = ARITHMETIC_4_NAMES,
        random_state: int = 42,
        engine: str = "cgp",
        engine_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.levels_back = levels_back
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.functions = functions
        self.function_names = function_names
        self.random_state = random_state
        self.engine = engine
        self.engine_params = engine_params

    def fit(self, X: Any, y: Any) -> "CGPClassifier":
        """Fits the Cartesian Genetic Programming classification model."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y)
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        _, n_features = X_arr.shape
        self.n_features_in_ = n_features

        loss_function = "bce" if self.n_classes_ == 2 else "cce"
        self.loss_function = loss_function
        num_outputs = 1 if self.n_classes_ == 2 else self.n_classes_

        self.interpreter_ = CartesianInterpreter(
            num_inputs=n_features,
            num_outputs=num_outputs,
            functions=self.functions,
            function_names=self.function_names,
        )

        self.fit_via_composer(
            X_arr,
            y_encoded,
            evaluator_cls=CartesianGPEvaluator,
            evaluator_config_cls=CartesianGPEvaluatorConfig,
        )

        logger.debug(
            "CGPClassifier fitted: best_fitness=%.6f, generations=%d",
            self.best_fitness_,
            self.max_generations,
        )

        return self

    def to_formula(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Returns the decoded symbolic mathematical expression for the classifier."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        names = feature_names or getattr(self, "feature_names_in_", None)
        return str(self.interpreter_.to_formula(self.best_genome_, feature_names=names))

    def formula(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Alias for to_formula."""
        return self.to_formula(feature_names=feature_names)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Estimates class probabilities for X."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        raw_preds = self.interpreter_.apply(self.best_genome_, jnp.array(X_arr))

        if self.n_classes_ == 2:
            probs_pos = np.asarray(jax.nn.sigmoid(raw_preds).squeeze())
            probs = np.column_stack([1.0 - probs_pos, probs_pos])
        else:
            probs = np.asarray(jax.nn.softmax(raw_preds, axis=-1))

        return probs

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for samples in X."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]

    def score(self, X: Any, y: Any) -> float:
        """Returns accuracy score on given test data and labels."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))
