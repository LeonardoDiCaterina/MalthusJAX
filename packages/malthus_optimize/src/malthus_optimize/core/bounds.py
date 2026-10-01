"""SciPy Bounds parser and transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ParsedBounds:
    """Standardized representation of parameter search bounds."""

    lower: np.ndarray
    upper: np.ndarray
    num_dims: int

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Scales physical parameters x from [lower, upper] to [0.0, 1.0]."""
        span = np.maximum(self.upper - self.lower, 1e-12)
        return (x - self.lower) / span

    def denormalize(self, norm_x: np.ndarray) -> np.ndarray:
        """Scales normalized parameters from [0.0, 1.0] back to [lower, upper]."""
        return self.lower + norm_x * (self.upper - self.lower)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clips physical parameters to [lower, upper]."""
        return np.clip(x, self.lower, self.upper)


def parse_bounds(
    bounds: Union[
        Sequence[Tuple[float, float]],
        ParsedBounds,
        Any,
    ],
    num_dims: int | None = None,
) -> ParsedBounds:
    """Parses various SciPy bound formats into a uniform ParsedBounds object.

    Supports:
    - Sequence of pairs: [(-5.0, 5.0), (-2.0, 2.0), ...]
    - Tuple of 2 arrays/lists: ([lb0, lb1, ...], [ub0, ub1, ...])
    - scipy.optimize.Bounds object (with .lb and .ub)
    - 2D array of shape (N, 2)
    - Single tuple (min, max) repeated across num_dims
    """
    if isinstance(bounds, ParsedBounds):
        return bounds

    # 1. scipy.optimize.Bounds object
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lb = np.asarray(bounds.lb, dtype=np.float32)
        ub = np.asarray(bounds.ub, dtype=np.float32)
        return ParsedBounds(lower=lb, upper=ub, num_dims=len(lb))

    # 2. 2D numpy array (N, 2)
    if isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] == 2:
        lb = bounds[:, 0].astype(np.float32)
        ub = bounds[:, 1].astype(np.float32)
        return ParsedBounds(lower=lb, upper=ub, num_dims=len(lb))

    if isinstance(bounds, (list, tuple)):
        # 3. List of (min, max) pairs: [(-5.0, 5.0), (-2.0, 2.0), ...]
        if isinstance(bounds, list) and all(
            isinstance(b, (list, tuple, np.ndarray)) and len(b) == 2 for b in bounds
        ):
            lowers = [float(b[0]) for b in bounds]
            uppers = [float(b[1]) for b in bounds]
            return ParsedBounds(
                lower=np.array(lowers, dtype=np.float32),
                upper=np.array(uppers, dtype=np.float32),
                num_dims=len(lowers),
            )

        # 4. SciPy curve_fit style 2-tuple: ([lb0, lb1, ...], [ub0, ub1, ...])
        if (
            len(bounds) == 2
            and isinstance(bounds[0], (list, tuple, np.ndarray))
            and isinstance(bounds[1], (list, tuple, np.ndarray))
        ):
            lb = np.asarray(bounds[0], dtype=np.float32)
            ub = np.asarray(bounds[1], dtype=np.float32)
            if len(lb) == len(ub):
                return ParsedBounds(lower=lb, upper=ub, num_dims=len(lb))

        # 5. Single (min, max) scalar pair applied to all dimensions
        if len(bounds) == 2 and not isinstance(bounds[0], (list, tuple, np.ndarray)):
            if num_dims is None:
                raise ValueError(
                    "num_dims must be specified when bounds is a single (min, max) pair"
                )
            lb = np.full(num_dims, bounds[0], dtype=np.float32)
            ub = np.full(num_dims, bounds[1], dtype=np.float32)
            return ParsedBounds(lower=lb, upper=ub, num_dims=num_dims)

        # 6. Tuple of (min, max) pairs
        if all(isinstance(b, (list, tuple, np.ndarray)) and len(b) == 2 for b in bounds):
            lowers = [float(b[0]) for b in bounds]
            uppers = [float(b[1]) for b in bounds]
            return ParsedBounds(
                lower=np.array(lowers, dtype=np.float32),
                upper=np.array(uppers, dtype=np.float32),
                num_dims=len(lowers),
            )

    raise TypeError(f"Unsupported bounds format: {type(bounds)}")


__all__ = ["ParsedBounds", "parse_bounds"]
