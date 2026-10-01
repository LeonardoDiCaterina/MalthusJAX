"""Scikit-Learn compatible Differentiable Cartesian Genetic Programming Regressor."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from malthus_gp.core.functions import ACTIVATION_FUNCTIONS, ACTIVATION_NAMES
from malthus_gp.evaluators.cartesian_gp_evaluator import (
    CartesianGPEvaluator,
    CartesianGPEvaluatorConfig,
)
from malthus_gp.evaluators.lsmf_evaluator import LSMFEvaluator, LSMFEvaluatorConfig
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)
from malthus_gp.learn.base import BaseGPEstimator, RegressorMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.learn.dcgp_regressor")


class DCGPRegressor(BaseGPEstimator, RegressorMixin):  # type: ignore[misc]
    """Differentiable Cartesian Genetic Programming Symbolic Regressor.

    Combines discrete evolutionary architecture search over Cartesian DAGs with
    continuous gradient descent (Adam/SGD) on edge weights and biases (LSMF
    memetic algorithm, Izzo et al., 2016; Märtens & Izzo, 2019).

    Parameters
    ----------
    n_rows : int, default=1
        Number of rows in the Cartesian grid.
    n_cols : int, default=30
        Number of columns in the Cartesian grid.
    levels_back : int, default=-1
        Connection locality window (-1 for full connectivity).
    pop_size : int, default=5
        Total population size (1 + λ).
    max_generations : int, default=200
        Maximum evolutionary generations.
    lsmf_epochs : int, default=10
        Number of inner gradient descent steps on continuous weights per evaluation.
    learning_rate : float, default=0.02
        Learning rate for the inner continuous optimizer (Adam).
    mutation_rate : float or None, default=None
        Per-gene topological mutation rate.
    activations : tuple of callables, default=ACTIVATION_FUNCTIONS
        Set of differentiable node activations.
    activation_names : tuple of str, default=ACTIVATION_NAMES
        Corresponding activation names for formula extraction.
    random_state : int, default=42
        PRNG seed.
    """

    def __init__(
        self,
        n_rows: int = 1,
        n_cols: int = 30,
        levels_back: int = -1,
        pop_size: int = 5,
        max_generations: int = 200,
        lsmf_epochs: int = 10,
        learning_rate: float = 0.02,
        mutation_rate: Optional[float] = None,
        activations: Tuple[Callable[[jax.Array], jax.Array], ...] = ACTIVATION_FUNCTIONS,
        activation_names: Tuple[str, ...] = ACTIVATION_NAMES,
        random_state: int = 42,
        engine: str = "cgp",
        engine_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.levels_back = levels_back
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.lsmf_epochs = lsmf_epochs
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.activations = activations
        self.activation_names = activation_names
        self.random_state = random_state
        self.engine = engine
        self.engine_params = engine_params

    def fit(self, X: Any, y: Any) -> "DCGPRegressor":
        """Fits the Differentiable Cartesian Genetic Programming model."""
        self.feature_names_in_: Optional[Sequence[str]] = (
            list(X.columns) if hasattr(X, "columns") else None
        )

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        _, n_features = X_arr.shape
        self.n_features_in_ = n_features

        self.interpreter_ = DifferentiableCartesianInterpreter(
            num_inputs=n_features,
            num_outputs=1,
            activations=self.activations,
            activation_names=self.activation_names,
        )

        base_eval_cfg = CartesianGPEvaluatorConfig(
            loss_function="mse",
            maximize=False,
        )
        base_evaluator = CartesianGPEvaluator(
            config=base_eval_cfg,
            data=(jnp.array(X_arr), jnp.array(y_arr)),
            interpreter=self.interpreter_,
        )

        lsmf_cfg = LSMFEvaluatorConfig(epochs=self.lsmf_epochs, maximize=False)
        evaluator = LSMFEvaluator(
            config=lsmf_cfg,
            data=base_evaluator.data,
            base_evaluator=base_evaluator,
            optimizer=optax.adam(self.learning_rate),
        )

        self.fit_via_composer(
            X,
            y,
            evaluator=evaluator,
        )

        logger.debug(
            "DCGPRegressor fitted: best_fitness=%.6f, generations=%d",
            self.best_fitness_,
            self.max_generations,
        )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts target values for X."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'predict'.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        preds_jax = self.interpreter_.apply(self.best_genome_, jnp.array(X_arr))
        return np.asarray(preds_jax).squeeze()

    def score(self, X: Any, y: Any) -> float:
        """Returns coefficient of determination R^2."""
        y_true = np.asarray(y, dtype=np.float32).squeeze()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def to_formula(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Returns the decoded mathematical formula with learned weights."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        names = feature_names or self.feature_names_in_
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
