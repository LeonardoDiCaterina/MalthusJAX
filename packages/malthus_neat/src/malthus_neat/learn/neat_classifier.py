"""Scikit-Learn compatible NEAT Classifier."""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from malthus_neat.learn.base import BaseNEATEstimator, ClassifierMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.learn.neat_classifier")


class NEATClassifier(BaseNEATEstimator, ClassifierMixin):  # type: ignore[misc]
    """Neuroevolution of Augmenting Topologies (NEAT) Classifier.

    Evolves neural networks for binary and multi-class classification tasks
    with cross-entropy loss metrics and probabilistic inferences.

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
    engine : str, default="neat"
        Execution engine name ("neat", "island_neat", "map_elites", "island_map_elites").
    engine_params : dict, optional
        Additional parameter overrides for the selected engine.
    """

    classes_: np.ndarray
    n_classes_: int

    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 100,
        max_nodes: int = 20,
        max_conns: int = 40,
        recurrent: bool = False,
        elitism: int = 2,
        random_state: int = 42,
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
        self.engine = engine
        self.engine_params = engine_params

    def fit(self, X: Any, y: Any) -> "NEATClassifier":
        """Fits the NEAT classification model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : NEATClassifier
            Fitted classifier instance.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = np.asarray(y)
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        loss_function = "bce" if self.n_classes_ == 2 else "cce"
        self.loss_function = loss_function
        num_outputs = 1 if self.n_classes_ == 2 else self.n_classes_

        y_targets = y_encoded.reshape(-1, 1) if num_outputs == 1 else y_encoded
        self.fit_via_composer(X_arr, y_targets)

        logger.debug(
            "NEATClassifier fitted: best_fitness=%.6f, generations=%d, classes=%d",
            self.best_fitness_,
            self.max_generations,
            self.n_classes_,
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Estimates class probability distributions for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Normalized class probabilities.
        """
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'predict_proba'.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        raw_preds = self.interpreter_.apply(self.best_genome_, jnp.array(X_arr))

        if self.n_classes_ == 2:
            probs_pos = np.asarray(jax.nn.sigmoid(raw_preds).squeeze())
            if probs_pos.ndim == 0:
                probs_pos = np.array([float(probs_pos)])
            probs = np.column_stack([1.0 - probs_pos, probs_pos])
        else:
            probs = np.asarray(jax.nn.softmax(raw_preds, axis=-1))

        return probs

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for X using the evolved best genome.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels matching original label types.
        """
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.asarray(self.classes_[pred_indices])

    def score(self, X: Any, y: Any) -> float:
        """Returns the mean classification accuracy on the given test data and labels."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))


__all__ = ["NEATClassifier"]
