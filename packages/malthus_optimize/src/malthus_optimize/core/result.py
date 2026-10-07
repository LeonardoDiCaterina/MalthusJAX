"""SciPy-compatible OptimizeResult container."""

from __future__ import annotations

from typing import Any

try:
    from scipy.optimize import OptimizeResult as _ScipyOptimizeResult
except ImportError:
    _ScipyOptimizeResult = dict  # type: ignore[misc,assignment]


class OptimizeResult(_ScipyOptimizeResult):
    """Container for the results of a mathematical optimization algorithm.

    Attributes:
        x (ndarray): The solution of the optimization.
        success (bool): Whether or not the optimizer exited successfully.
        status (int): Termination status of the optimizer.
        message (str): Description of the cause of the termination.
        fun (float): Value of the objective function at the solution.
        nfev (int): Number of evaluations of the objective functions.
        nit (int): Number of iterations performed by the optimizer.
        population (ndarray, optional): Final population parameter matrix.
        population_energies (ndarray, optional): Final population objective values.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'OptimizeResult' object has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'OptimizeResult' object has no attribute '{name}'") from None

    def __repr__(self) -> str:
        items = [f" {k}: {v!r}" for k, v in sorted(self.items())]
        return "message: " + str(self.get("message", "")) + "\n" + "\n".join(items)


__all__ = ["OptimizeResult"]
