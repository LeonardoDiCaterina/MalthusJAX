"""Scikit-Learn compatible NEAT Regressor."""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from malthus_neat.learn.base import BaseNEATEstimator, RegressorMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.learn.neat_regressor")


class NEATRegressor(BaseNEATEstimator, RegressorMixin):  # type: ignore[misc]
    """Neuroevolution of Augmenting Topologies (NEAT) Symbolic Regressor.

    Evolves neural network topologies and continuous synaptic weights using
    vectorized JAX kernels and the MalthusJAX engine catalog.

    Parameters
    ----------
    pop_size : int, default=50
        Population size per generation.
    max_generations : int, default=100
        Maximum evolutionary generations.
    max_nodes : int, default=20
        Maximum node capacity in static XLA memory buffers.
    max_conns : int, default=40
        Maximum synapse capacity in static XLA memory buffers.
    recurrent : bool, default=False
        Whether feedback/recurrent loops are permitted.
    elitism : int, default=2
        Number of top individuals preserved per generation.
    random_state : int, default=42
        Seed for the JAX PRNG entropy source.
    loss_function : str, default="mse"
        Loss metric for optimization ("mse" or "mae").
    engine : str, default="neat"
        Execution engine name ("neat", "island_neat", "map_elites", "island_map_elites").
    engine_params : dict, optional
        Additional parameter overrides for the selected engine.
    """

    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 100,
        max_nodes: int = 20,
        max_conns: int = 40,
        recurrent: bool = False,
        elitism: int = 2,
        random_state: int = 42,
        loss_function: str = "mse",
        engine: str = "neat",
        engine_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.recurrent = recurrent
        self.elitism = elitism
        self.random_state = random_state
        self.loss_function = loss_function
        self.engine = engine
        self.engine_params = engine_params

    def fit(self, X: Any, y: Any) -> "NEATRegressor":
        """Fits the NEAT neural network on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target scalar values.

        Returns
        -------
        self : NEATRegressor
            Fitted estimator instance.
        """
        self.fit_via_composer(X, y)
        logger.debug(
            "NEATRegressor fitted: best_fitness=%.6f, generations=%d",
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
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
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


__all__ = ["NEATRegressor"]
