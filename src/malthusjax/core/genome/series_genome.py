"""Series genome: function-valued genomes via spectral coefficient arrays.

Provides a genome representation where each individual encodes a multi-output
function as a 2-D coefficient matrix:

    values : (n_dims, n_coeffs)

Row ``i`` holds the ``n_coeffs`` spectral coefficients for output dimension
``i``, interpreted by a :class:`BasisFunction`.  Expanding the genome shape
from 1-D (:class:`~malthusjax.core.genome.real_genome.RealGenome`) to 2-D
here directly addresses the ``TODO`` in ``RealPopulation`` and unlocks:

* **ODE fitting** — one genome simultaneously encodes every state variable,
  enabling physics-informed fitness via :meth:`SeriesGenome.eval_deriv`.
* **PDE fitting** — rows index spatial modes for separable expansions
  ``u(x, t) = sum_i a_i(t) phi_i(x)``.
* **Operator learning** — rows parameterise a function-to-function mapping.

Three built-in bases are provided: :class:`FourierBasis`,
:class:`ChebyshevBasis`, and :class:`MonomialBasis`.  Custom bases are added
by subclassing :class:`BasisFunction`.

All basis ``eval`` methods are fully JAX-traceable and compose cleanly with
``jax.vmap``, ``jax.jit``, and ``jax.grad``.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.base import DistanceMetric as BaseDistances

_field: Any = struct.field


class BasisFunction(ABC):
    """Abstract spectral basis: maps ``(t, coeffs)`` → scalar value.

    A concrete ``BasisFunction`` defines how a 1-D coefficient row should be
    interpreted as a scalar function of one variable.  It is stored inside
    :class:`SeriesGenomeConfig` as a ``pytree_node=False`` field, making it
    a compile-time constant from JAX's perspective — exactly like ``dtype`` in
    :class:`~malthusjax.core.genome.real_genome.RealGenomeConfig`.

    Contract
    --------
    * :meth:`eval` **must** be JAX-traceable with no Python branches on
      traced values.  Use ``jnp.where`` instead of ``if`` over arrays.
    * :meth:`eval_deriv` has a default implementation via ``jax.grad`` that
      works for any differentiable :meth:`eval`.  Override only for bases
      whose analytic derivative is cheaper to compute (e.g. Fourier).

    Parameters passed to :meth:`eval`
    -----------------------------------
    t      : scalar float — the independent variable (e.g. time).
    coeffs : 1-D float array of shape ``(n_coeffs,)`` — one row of the genome.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short string identifier, used in TOML and catalog registration."""
        ...

    @abstractmethod
    def eval(self, t: float, coeffs: chex.Array) -> chex.Array:
        """Evaluate the series at a single point ``t``.

        Parameters
        ----------
        t      : Scalar time / independent variable.
        coeffs : 1-D array of shape ``(n_coeffs,)`` — coefficients for **one**
                 output dimension.

        Returns
        -------
        Array scalar — the function value at ``t``.
        """
        ...

    def eval_deriv(self, t: float, coeffs: chex.Array) -> chex.Array:
        """First derivative d/dt of :meth:`eval` at ``t``.

        Default implementation uses ``jax.grad``.  Subclasses may override
        with an analytic derivative for efficiency.

        Parameters
        ----------
        t      : Scalar time.
        coeffs : 1-D array of shape ``(n_coeffs,)``.

        Returns
        -------
        Array scalar — the derivative at ``t``.
        """
        return jax.grad(self.eval, argnums=0)(t, coeffs)

    def eval_deriv_nth(self, t: float, coeffs: chex.Array, n: int) -> chex.Array:
        """n-th derivative d^n/dt^n of :meth:`eval` at ``t``.

        Uses repeated application of ``jax.grad``.  Subclasses may override
        with an analytic implementation for efficiency.

        Parameters
        ----------
        t      : Scalar time.
        coeffs : 1-D array of shape ``(n_coeffs,)``.
        n      : Order of the derivative (positive integer).

        Returns
        -------
        float scalar — the n-th derivative at ``t``.
        """
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n}.")
        deriv_fn = self.eval
        for _ in range(n):
            deriv_fn = jax.grad(deriv_fn, argnums=0)
        return deriv_fn(t, coeffs)


class FourierBasis(BasisFunction):
    """Truncated Fourier series on the interval ``[t0, t1]``.

    .. math::

        f(t) = a_0 + \\sum_{n=1}^{N}
               \\bigl[ a_n \\cos(n \\omega t) + b_n \\sin(n \\omega t) \\bigr],
        \\quad \\omega = \\frac{2\\pi}{t_1 - t_0}

    Coefficient layout in one genome row::

        coeffs = [a0, a1, b1, a2, b2, ..., aN, bN]

    **Constraint**: ``n_coeffs`` must be **odd** (``1 + 2 * n_pairs``).
    Call :meth:`SeriesGenomeConfig.validate` to check this before training.

    Parameters
    ----------
    t0, t1 : float
        Domain endpoints.  Must satisfy ``t0 < t1``.
    """

    def __init__(self, t0: float = 0.0, t1: float = 1.0) -> None:
        self.t0 = float(t0)
        self.t1 = float(t1)

    @property
    def name(self) -> str:
        return "fourier"

    def eval(self, t: float, coeffs: chex.Array) -> chex.Array:
        a0 = coeffs[0]
        pairs = coeffs[1:].reshape(-1, 2)  # (n_pairs, 2)
        n = jnp.arange(1, pairs.shape[0] + 1, dtype=jnp.float32)
        w = 2.0 * jnp.pi / (self.t1 - self.t0)
        return a0 + jnp.sum(pairs[:, 0] * jnp.cos(n * w * t) + pairs[:, 1] * jnp.sin(n * w * t))

    def eval_deriv(self, t: float, coeffs: chex.Array) -> chex.Array:
        """Analytic derivative — avoids a second pass through the Fourier eval."""
        pairs = coeffs[1:].reshape(-1, 2)  # (n_pairs, 2)
        n = jnp.arange(1, pairs.shape[0] + 1, dtype=jnp.float32)
        w = 2.0 * jnp.pi / (self.t1 - self.t0)
        nw = n * w
        return jnp.sum(-pairs[:, 0] * nw * jnp.sin(nw * t) + pairs[:, 1] * nw * jnp.cos(nw * t))


class ChebyshevBasis(BasisFunction):
    """Chebyshev polynomial expansion on ``[t0, t1]``.

    .. math::

        f(t) = \\sum_{n=0}^{N-1} c_n \\, T_n(x(t)),
        \\quad x(t) = \\frac{2(t - t_0)}{t_1 - t_0} - 1

    where :math:`T_n` are Chebyshev polynomials of the first kind, evaluated
    via the three-term recurrence

    .. math::

        T_0 = 1, \\quad T_1 = x, \\quad T_n = 2x \\, T_{n-1} - T_{n-2}

    implemented with :func:`jax.lax.scan` (branch-free, JIT-stable).

    Compared to :class:`FourierBasis`, Chebyshev polynomials are orthogonal
    on ``[-1, 1]`` with a non-uniform weight, which makes them better
    conditioned for approximating smooth functions with isolated features.

    Parameters
    ----------
    t0, t1 : float
        Domain endpoints.
    """

    def __init__(self, t0: float = 0.0, t1: float = 1.0) -> None:
        self.t0 = float(t0)
        self.t1 = float(t1)

    @property
    def name(self) -> str:
        return "chebyshev"

    @staticmethod
    def _basis_values(x: float, n: int) -> chex.Array:
        """Compute ``[T_0(x), T_1(x), ..., T_{n-1}(x)]`` via recurrence.

        Uses ``jax.lax.scan`` with the carry ``(T_{k}, T_{k+1})``, emitting
        ``T_k`` at each step.  The ``length`` argument is a static Python int
        (derived from ``coeffs.shape[0]``) so JAX can unroll or compile it.

        Parameters
        ----------
        x : Mapped domain value in ``[-1, 1]``.
        n : Number of basis values to compute (== ``n_coeffs``).

        Returns
        -------
        Array of shape ``(n,)`` with values ``[T_0, T_1, ..., T_{n-1}]``.
        """
        x_arr = jnp.asarray(x)

        def step(
            carry: tuple[chex.Array, chex.Array], _: Any
        ) -> tuple[tuple[chex.Array, chex.Array], chex.Array]:
            t_prev, t_curr = carry
            t_next = 2.0 * x_arr * t_curr - t_prev
            return (t_curr, t_next), t_prev  # emit t_prev; advance carry

        _, values = jax.lax.scan(
            step,
            init=(jnp.ones_like(x_arr), x_arr),  # carry = (T_0 = 1, T_1 = x)
            xs=None,
            length=n,
        )
        return values  # shape (n,) = [T_0, ..., T_{n-1}]

    def eval(self, t: float, coeffs: chex.Array) -> chex.Array:
        x = 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0
        basis = self._basis_values(x, coeffs.shape[0])
        return jnp.dot(coeffs, basis)


class MonomialBasis(BasisFunction):
    """Power series (monomial) basis.

    .. math::

        f(t) = \\sum_{n=0}^{N-1} c_n \\, t^n

    Evaluated via **Horner's method** for improved numerical stability:

    .. math::

        f(t) = c_0 + t \\bigl(c_1 + t(c_2 + \\cdots + t \\, c_{N-1})\\bigr)

    implemented with :func:`jax.lax.scan`.

    .. warning::

        High-degree monomials can become poorly conditioned for large ``|t|``
        or ``n_coeffs``.  Prefer :class:`ChebyshevBasis` when fitting smooth
        functions, or restrict the domain to a small interval around zero.
    """

    @property
    def name(self) -> str:
        return "monomial"

    def eval(self, t: float, coeffs: chex.Array) -> chex.Array:
        # Horner sweep from highest to lowest degree coefficient.
        # reversed_coeffs = [c_{n-1}, c_{n-2}, ..., c_0]
        reversed_coeffs = jnp.flip(coeffs)

        def step(acc: chex.Array, c: chex.Array) -> tuple[chex.Array, Any]:
            return acc * t + c, None

        result, _ = jax.lax.scan(
            step,
            init=jnp.zeros((), dtype=coeffs.dtype),
            xs=reversed_coeffs,
        )
        return result


class SeriesDistanceMetric(BaseDistances):
    """Distance metrics for 2-D coefficient matrices.

    Extends the base metric names with ``FROBENIUS`` (the natural 2-D
    generalisation of Euclidean distance).
    """

    FROBENIUS: str = "frobenius"  # sqrt(sum of squared element-wise differences)
    # Inherits: EUCLIDEAN (alias for frobenius), MANHATTAN, HAMMING


@struct.dataclass
class SeriesGenomeConfig:
    """Static configuration for series-valued genomes.

    All fields are ``pytree_node=False`` — they are compile-time constants
    from JAX's perspective, exactly as in
    :class:`~malthusjax.core.genome.real_genome.RealGenomeConfig`.

    Parameters
    ----------
    n_dims : int
        Number of output dimensions (= number of coefficient rows).
        Examples:
        - ``n_dims=1`` — scalar function, equivalent to 1-D Fourier fitting
        - ``n_dims=2`` — 2-state ODE (e.g. pendulum: theta, omega)
        - ``n_dims=3`` — 3-D vector field, Lorenz system, etc.

    n_coeffs : int
        Number of spectral coefficients **per dimension**.
        For :class:`FourierBasis` this must be **odd**: ``1 + 2 * n_pairs``.
        For :class:`ChebyshevBasis` and :class:`MonomialBasis` any positive
        integer is valid.

    basis : BasisFunction
        Interpretation of the coefficient rows.  Stored as a static Python
        object; changing this triggers JAX recompilation.

    bounds : tuple[float, float]
        Uniform ``[lower, upper]`` range applied to all coefficients during
        random initialisation.  **Not enforced after mutation** — see the
        :meth:`SeriesGenome.autocorrect` note.

    dtype : jax dtype
        Floating-point precision.  Defaults to ``jnp.float32``.

    Notes
    -----
    Call :meth:`validate` before starting any optimisation run to catch
    configuration errors early (e.g. even ``n_coeffs`` with FourierBasis).
    """

    n_dims: int = _field(pytree_node=False, default=1)
    n_coeffs: int = _field(pytree_node=False, default=7)
    basis: BasisFunction = _field(pytree_node=False, default_factory=lambda: FourierBasis())
    bounds: Tuple[float, float] = _field(pytree_node=False, default=(-2.0, 2.0))
    dtype: Any = _field(pytree_node=False, default=jnp.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        """Logical shape of a single genome's coefficient array."""
        return (self.n_dims, self.n_coeffs)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is self-inconsistent.

        Checks:
        * ``n_dims >= 1`` and ``n_coeffs >= 1``.
        * ``bounds`` satisfies ``lower < upper``.
        * ``FourierBasis`` requires odd ``n_coeffs``.

        Call this once before training, not inside JIT-compiled code.
        """
        if self.n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {self.n_dims}.")
        if self.n_coeffs < 1:
            raise ValueError(f"n_coeffs must be >= 1, got {self.n_coeffs}.")
        lo, hi = self.bounds
        if lo >= hi:
            raise ValueError(f"bounds must satisfy lower < upper, got {self.bounds}.")
        if isinstance(self.basis, FourierBasis) and self.n_coeffs % 2 == 0:
            odd_up = self.n_coeffs + 1
            odd_down = self.n_coeffs - 1
            raise ValueError(
                f"FourierBasis requires odd n_coeffs (1 + 2*n_pairs), "
                f"got {self.n_coeffs}.  Try {odd_down} or {odd_up}."
            )

    def init_population(self, key: chex.PRNGKey, size: int) -> "SeriesPopulation":
        """Create a random population from this config (protocol method)."""
        return SeriesPopulation.init_random(key, self, size)


@struct.dataclass
class SeriesGenome(BaseGenome):
    """A genome that encodes a multi-output function via spectral coefficients.

    **Shape semantics**:

    +------------------+--------------------------------+
    | Context          | ``values`` shape               |
    +==================+================================+
    | Single genome    | ``(n_dims, n_coeffs)``         |
    +------------------+--------------------------------+
    | Batched (SoA)    | ``(pop_size, n_dims, n_coeffs)``|
    +------------------+--------------------------------+

    This is the direct 2-D generalisation of :class:`RealGenome`, which uses
    shape ``(n,)`` for single and ``(pop_size, n)`` for batched.

    Calling :meth:`eval` or :meth:`eval_deriv` on a *single* genome (inside
    ``jax.vmap`` over the population) is the intended usage pattern; see the
    evaluator for an example.
    """

    values: chex.Array
    subscriptable: bool = _field(pytree_node=False, default=True)

    # ------------------------------------------------------------------ #
    # BaseGenome protocol
    # ------------------------------------------------------------------ #

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: SeriesGenomeConfig) -> "SeriesGenome":
        """Sample a genome uniformly within ``config.bounds``.

        Each coefficient is drawn independently from
        ``Uniform(bounds[0], bounds[1])``.
        """
        lo, hi = config.bounds
        values = jax.random.uniform(key, config.shape, minval=lo, maxval=hi, dtype=config.dtype)
        return cls(values=values)

    def autocorrect(self, config: SeriesGenomeConfig) -> "SeriesGenome":
        """Clamp all coefficients into ``[config.bounds[0], config.bounds[1]]``.

        Mirrors :meth:`~malthusjax.core.genome.real_genome.RealGenome.autocorrect`.
        Bounds are **not** enforced automatically after mutation; call this
        explicitly or enable ``clip=True`` in the mutation operator.
        """
        lo, hi = config.bounds
        return dataclasses.replace(self, values=jnp.clip(self.values, lo, hi))

    def distance(
        self,
        other: BaseGenome,
        metric: str = SeriesDistanceMetric.FROBENIUS,
    ) -> chex.Numeric:
        """Distance between two coefficient matrices.

        Parameters
        ----------
        other  : Another :class:`SeriesGenome` (cast internally).
        metric : One of ``"frobenius"`` / ``"euclidean"`` (identical),
                 ``"manhattan"``, ``"hamming"``.

        Notes
        -----
        ``"frobenius"`` and ``"euclidean"`` are identical for flat arrays;
        both compute :math:`\\|A - B\\|_F = \\sqrt{\\sum_{ij}(A_{ij}-B_{ij})^2}`.
        """
        diff = self.values - cast("SeriesGenome", other).values

        if metric in (SeriesDistanceMetric.FROBENIUS, SeriesDistanceMetric.EUCLIDEAN):
            return jnp.sqrt(jnp.sum(jnp.square(diff)))
        elif metric == SeriesDistanceMetric.MANHATTAN:
            return jnp.sum(jnp.abs(diff))
        elif metric == SeriesDistanceMetric.HAMMING:
            value_range = jnp.max(jnp.abs(self.values)) + 1e-8
            threshold = 0.01 * value_range
            return jnp.sum(jnp.abs(diff) > threshold)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

    @property
    def size(self) -> int:
        """Total number of coefficients: ``n_dims * n_coeffs``."""
        return int(self.values.shape[-2] * self.values.shape[-1])

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape of ``values``.

        - Single genome: ``(n_dims, n_coeffs)``
        - Batched (SoA): ``(pop_size, n_dims, n_coeffs)``
        """
        return cast(Tuple[int, ...], self.values.shape)

    @classmethod
    def from_tensor(cls, arr: chex.Array, config: Any = None) -> "SeriesGenome":
        """Wrap a raw array as a :class:`SeriesGenome`.

        Intentionally minimal — no validation — to remain JIT-traceable.
        Expected shapes:

        * ``(pop_size, n_dims, n_coeffs)`` when called from
          :meth:`BasePopulation.from_array`.
        * ``(n_dims, n_coeffs)`` for a single individual.
        """
        return cls(values=arr)

    # ------------------------------------------------------------------ #
    # Algebraic helpers  (mirrors RealGenome.magnitude / normalize)
    # ------------------------------------------------------------------ #

    def magnitude(self) -> chex.Numeric:
        """Frobenius norm of the coefficient matrix."""
        return jnp.sqrt(jnp.sum(jnp.square(self.values)))

    def normalize(self) -> "SeriesGenome":
        """Scale the coefficient matrix to Frobenius norm 1.

        Uses ``jnp.where`` to avoid division by zero under JIT.
        """
        norm = self.magnitude()
        norm_safe = jnp.maximum(norm, 1e-8)
        normed = jnp.where(norm > 0, self.values / norm_safe, self.values)
        return dataclasses.replace(self, values=normed)

    # ------------------------------------------------------------------ #
    # Function evaluation  — the key capability added by SeriesGenome
    # ------------------------------------------------------------------ #

    def eval(self, t: float, config: SeriesGenomeConfig) -> chex.Array:
        """Evaluate the encoded function at a single point ``t``.

        Applies ``config.basis.eval(t, row)`` independently to each row of
        the coefficient matrix via ``jax.vmap``, returning one scalar per
        output dimension.

        Parameters
        ----------
        t      : Scalar independent variable (e.g. time).
        config : Genome config carrying the :class:`BasisFunction`.

        Returns
        -------
        Array of shape ``(n_dims,)`` — the function value at ``t``.

        Example
        -------
        >>> state = genome.eval(t, config)   # [theta(t), omega(t)]
        """
        return jax.vmap(config.basis.eval, in_axes=(None, 0))(t, self.values)

    def eval_deriv(self, t: float, config: SeriesGenomeConfig) -> chex.Array:
        """First derivative d/dt of :meth:`eval` at ``t``.

        Uses ``jax.jvp`` with tangent ``1.0`` to compute the full Jacobian
        column ``d(eval)/dt`` in one forward pass — no loops over dimensions.

        Parameters
        ----------
        t      : Scalar independent variable.
        config : Genome config.

        Returns
        -------
        Array of shape ``(n_dims,)`` — the time derivatives at ``t``.

        Example
        -------
        >>> dy_dt = genome.eval_deriv(t, config)   # [dtheta/dt, domega/dt]

        Notes
        -----
        ``jax.jvp`` is preferred over ``jax.grad`` here because :meth:`eval`
        returns a vector (not a scalar).  It is also preferred over
        ``jax.jacfwd`` because only the single tangent direction ``dt=1.0``
        is needed.
        """
        _, deriv = jax.jvp(
            lambda t_: self.eval(t_, config),
            (t,),
            (jnp.ones_like(t),),
        )
        return deriv


# =========================================================================== #
# Population
# =========================================================================== #


@struct.dataclass
class SeriesPopulation(BasePopulation[SeriesGenome]):
    """Container for a population of :class:`SeriesGenome` individuals.

    Implements the Struct-of-Arrays (SoA) pattern: ``genes.values`` has shape
    ``(pop_size, n_dims, n_coeffs)`` — a 3-D array that directly resolves the
    ``TODO`` in ``RealPopulation`` about supporting non-1-D genome shapes.

    Attributes
    ----------
    genes   : Batched :class:`SeriesGenome` with 3-D ``values``.
    fitness : Array of shape ``(pop_size,)`` — one scalar per individual.
    config  : Shared :class:`SeriesGenomeConfig` (static, not a pytree leaf).
    """

    genes: SeriesGenome
    fitness: chex.Array
    config: SeriesGenomeConfig = _field(pytree_node=False)

    GENOME_CLS: ClassVar[Type[SeriesGenome]] = SeriesGenome

    @classmethod
    def init_random(
        cls,
        key: chex.PRNGKey,
        config: SeriesGenomeConfig,
        size: int,
    ) -> "SeriesPopulation":
        """Create ``size`` random individuals, each sampled via :meth:`SeriesGenome.random_init`.

        Delegates to :meth:`BaseGenome.create_population` which vmaps
        ``random_init`` over split keys — no Python loop.
        """
        batched_genes = SeriesGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
