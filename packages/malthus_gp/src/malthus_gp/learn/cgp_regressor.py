"""Scikit-Learn compatible Cartesian Genetic Programming Regressor."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from malthus_gp.core.functions import ARITHMETIC_4_FUNCTIONS, ARITHMETIC_4_NAMES
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthus_gp.learn.base import BaseGPEstimator, RegressorMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.learn.cgp_regressor")


class CGPRegressor(BaseGPEstimator, RegressorMixin):  # type: ignore[misc]
    """Cartesian Genetic Programming Symbolic Regressor.

    Evolves feed-forward DAG symbolic regression models using the (1 + λ)-ES
    strategy with neutral drift (Miller & Thomson, 2000).

    Parameters
    ----------
    n_rows : int, default=1
        Number of rows in the Cartesian grid (1 for standard 1D linear DAG).
    n_cols : int, default=40
        Number of columns / computational nodes in the Cartesian grid.
    levels_back : int, default=-1
        Connection locality window (-1 for unconstrained full feed-forward connectivity).
    pop_size : int, default=5
        Total population size (1 + λ). Default 5 corresponds to 1 parent + 4 offspring.
    max_generations : int, default=2000
        Maximum evolutionary generations.
    mutation_rate : float or None, default=None
        Per-gene point mutation rate. If None, targets ~3 mutated genes per chromosome.
    functions : tuple of callables, default=ARITHMETIC_4_FUNCTIONS
        Tuple of primitive mathematical functions (+, -, *, /).
    function_names : tuple of str, default=ARITHMETIC_4_NAMES
        Corresponding string symbols for formula decoding.
    random_state : int, default=42
        Seed for the JAX PRNG entropy source.
    loss_function : str, default="mse"
        Loss metric for optimization ("mse" or "mae").
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
        loss_function: str = "mse",
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
        self.loss_function = loss_function
        self.engine = engine
        self.engine_params = engine_params

    def fit(self, X: Any, y: Any) -> "CGPRegressor":
        """Fits the Cartesian Genetic Programming model on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature vectors.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target scalar values.

        Returns
        -------
        self : CGPRegressor
            Fitted estimator instance.
        """
        self.feature_names_in_: Optional[Sequence[str]] = (
            list(X.columns) if hasattr(X, "columns") else None
        )

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        _, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.interpreter_ = CartesianInterpreter(
            num_inputs=n_features,
            num_outputs=1,
            functions=self.functions,
            function_names=self.function_names,
        )

        self.fit_via_composer(X, y)

        logger.debug(
            "CGPRegressor fitted: best_fitness=%.6f, generations=%d",
            self.best_fitness_,
            self.max_generations,
        )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts regression targets for X using the evolved best genome.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Model predictions.
        """
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'predict'.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        preds_jax = self.interpreter_.apply(self.best_genome_, jnp.array(X_arr))
        return np.asarray(preds_jax).squeeze()

    def score(self, X: Any, y: Any) -> float:
        """Returns the coefficient of determination R^2 of the prediction."""
        y_true = np.asarray(y, dtype=np.float32).squeeze()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def to_formula(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Returns the decoded symbolic mathematical expression for the model."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        names = feature_names or getattr(self, "feature_names_in_", None)
        return str(self.interpreter_.to_formula(self.best_genome_, feature_names=names))

    def formula(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Alias for to_formula."""
        return self.to_formula(feature_names=feature_names)

    def get_active_mask(self) -> np.ndarray:
        """Returns the boolean array indicating active computational nodes."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        mask = self.best_genome_.get_active_mask(self.n_features_in_)
        return np.asarray(mask)
