"""Automated Symbolic Feature Transformer.

Synthesizes informative, non-linear interaction features using Cartesian
Genetic Programming, maximizing correlation with the target while minimizing
feature redundancy (multicollinearity).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from malthus_gp.core.functions import ARITHMETIC_4_FUNCTIONS, ARITHMETIC_4_NAMES
from malthus_gp.evaluators.correlation_evaluator import (
    CorrelationEvaluator,
    CorrelationEvaluatorConfig,
)
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthus_gp.learn.base import BaseGPEstimator, TransformerMixin
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.learn.symbolic_transformer")


class SymbolicFeatureTransformer(BaseGPEstimator, TransformerMixin):  # type: ignore[misc]
    """Automated Symbolic Feature Transformer using Cartesian Genetic Programming.

    Evolves multi-output non-linear mathematical expressions to discover
    informative features from input data X. Features are optimized to maximize
    their correlation with the target variable y while penalizing pairwise
    cross-correlation (redundancy).

    Parameters
    ----------
    n_features : int, default=5
        Number of symbolic features to generate.
    include_original : bool, default=False
        If True, the transformed output contains the original features followed
        by the newly engineered symbolic features: [X_orig, X_symbolic].
    redundancy_weight : float, default=0.5
        Penalty weight for pairwise cross-correlation between the generated features.
    n_rows : int, default=1
        Number of rows in the Cartesian DAG grid.
    n_cols : int, default=30
        Number of columns in the Cartesian DAG grid.
    levels_back : int, default=-1
        Connection locality window (-1 for unrestricted feed-forward DAG).
    pop_size : int, default=5
        Population size (1 + λ). Default 5 implies λ=4.
    max_generations : int, default=150
        Maximum evolutionary generations.
    mutation_rate : float or None, default=None
        Per-gene point mutation rate.
    functions : tuple of callables, default=BASIC_FUNCTIONS
        Primitive operations (+, -, *, /).
    function_names : tuple of str, default=BASIC_FUNCTION_NAMES
        Names of primitive operations.
    random_state : int, default=42
        PRNG seed for reproducible evolution.
    """

    def __init__(
        self,
        n_features: int = 5,
        include_original: bool = False,
        redundancy_weight: float = 0.5,
        n_rows: int = 1,
        n_cols: int = 30,
        levels_back: int = -1,
        pop_size: int = 5,
        max_generations: int = 150,
        mutation_rate: Optional[float] = None,
        functions: Tuple[Callable[[jax.Array, jax.Array], jax.Array], ...] = ARITHMETIC_4_FUNCTIONS,
        function_names: Tuple[str, ...] = ARITHMETIC_4_NAMES,
        random_state: int = 42,
        engine: str = "cgp",
        engine_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_features = n_features
        self.include_original = include_original
        self.redundancy_weight = redundancy_weight
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

    def fit(self, X: Any, y: Any = None) -> "SymbolicFeatureTransformer":
        """Fits the symbolic transformer on training data X and target y."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.n_features_in_ = X_arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(self.n_features_in_)]

        # If y is not provided, use an unsupervised target (linear sum of inputs)
        if y is None:
            y_arr = np.sum(X_arr, axis=1, dtype=np.float32)
        else:
            y_arr = np.asarray(y, dtype=np.float32).flatten()

        self.interpreter_ = CartesianInterpreter(
            functions=self.functions,
            function_names=self.function_names,
            num_inputs=self.n_features_in_,
            num_outputs=self.n_features,
        )

        self.fit_via_composer(
            X_arr,
            y_arr,
            evaluator_cls=CorrelationEvaluator,
            evaluator_config_cls=CorrelationEvaluatorConfig,
            redundancy_weight=self.redundancy_weight,
        )

        logger.debug(
            "SymbolicFeatureTransformer fitted: best_fitness=%.6f, n_features=%d",
            self.best_fitness_,
            self.n_features,
        )

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transforms input data X into engineered symbolic features."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' before 'transform'.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_jax = jnp.array(X_arr)
        preds_jax = self.interpreter_.apply(self.best_genome_, X_jax)
        X_trans = np.asarray(preds_jax)
        if X_trans.ndim == 1:
            X_trans = X_trans.reshape(-1, 1)

        if self.include_original:
            return np.hstack([X_arr, X_trans])
        return X_trans

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        """Get output feature names for transformation."""
        feat_in = getattr(self, "feature_names_in_", None)
        if input_features is not None:
            in_names = list(input_features)
        elif feat_in is not None:
            in_names = list(feat_in)
        else:
            in_names = [f"x{i}" for i in range(getattr(self, "n_features_in_", 1))]
        out_names = [f"cgp_feat_{i}" for i in range(self.n_features)]
        if self.include_original:
            return in_names + out_names
        return out_names

    def to_formula(self, feature_names: Optional[Sequence[str]] = None) -> List[str]:
        """Returns the list of mathematical formulas for all generated features."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")
        names = feature_names or self.feature_names_in_
        return [
            self.interpreter_.to_formula(self.best_genome_, feature_names=names, output_idx=i)
            for i in range(self.n_features)
        ]

    def formula(self, feature_names: Optional[Sequence[str]] = None) -> List[str]:
        """Alias for to_formula."""
        return self.to_formula(feature_names=feature_names)
