"""Base interfaces for the composable evaluator architecture.

Defines the three-axis decomposition:
  Problem  = TaskType × Environment × Interpreter
  Output   = ScalarOutput | QDOutput(DescriptorFn) | MOOutput
  Evaluator = Problem × Output

See docs/evaluator_design.md for the full design document.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import chex
import jax.numpy as jnp
import jax

from flax import struct

from malthusjax.core.base import BaseGenome

G = TypeVar("G", bound=BaseGenome)


# =============================================================================
# Transforms
# =============================================================================

@struct.dataclass
class BaseTransform(Generic[G]):
    """Genotype to phenotype mapping axis."""
    def transform(self, genome: G, state: Any = None) -> Any:
        """Convert a single genome to a computationally optimized representation."""
        return genome
        
    def transform_population(self, genes: Any, state: Any = None) -> Any:
        """Convert a population of genomes. Overridable for population-level transforms."""
        return jax.vmap(self.transform, in_axes=(0, None))(genes, state)

@struct.dataclass
class IdentityTransform(BaseTransform[Any]):
    """Default no-op transform for direct parameter encoding."""
    pass


# =============================================================================
# BaseInterpreter
# =============================================================================

@struct.dataclass
class BaseInterpreter(Generic[G]):
    """Decodes a genome into outputs given inputs.

    The Interpreter is the genome-side bridge. It is:
    - **Stateless**: all configuration (dims, architecture) is compiled in at
      construction time as pytree_node=False fields.
    - **Genome-specific**: typed to a specific genome type ``G``.
    - **Task-agnostic**: its output is always ``apply(genome, inputs) → outputs``.

    For native MalthusJAX genomes (RealGenome, LinearGenome, etc.), the user
    declares the Interpreter explicitly.

    For adapter frameworks (TensorNEAT, evosax), the adapter provides an
    Interpreter via ``adapter.interpreter`` — the user never constructs it.

    The key invariant::

        genome_spec.length == interpreter.num_params

    The Composer validates this at build time and raises ``ConfigurationError``
    loudly if there is a mismatch, before any JAX tracing begins.
    """

    def apply(self, genome: G, inputs: chex.Array) -> chex.Array:
        """Apply the genome as a function over the given inputs.

        Args:
            genome: A single (unbatched) genome.
            inputs: The input array — could be an observation vector (RL),
                a data matrix row (SL), or None (Optimization tasks).

        Returns:
            Output array — action logits (RL), predictions (SL), or raw
            solution vector (Optimization, via IdentityInterpreter).
        """
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        """Total number of genome values consumed by this interpreter.

        Returns -1 for ``IdentityInterpreter`` where the genome length is
        dictated by the problem, not the interpreter.
        """
        raise NotImplementedError


# =============================================================================
# BaseEnvironment hierarchy
# =============================================================================

@struct.dataclass
class BaseEnvironment:
    """Root base class for all environments.

    Concrete environments inherit from one of the three task-type-specific
    subclasses:  ``BaseOptimizationEnvironment``, ``BaseSupervisedEnvironment``,
    or ``BaseRLEnvironment``.

    The Task Type is fixed by the environment: a ``SklearnEnv`` is always
    a SupervisedTask; a ``BBOBEnv`` is always an OptimizationTask.
    Users never declare the Task Type explicitly.
    """
    pass


@struct.dataclass
class BaseOptimizationEnvironment(BaseEnvironment):
    """Environment for OptimizationTask: f(genome, instance_data) → scalar.

    Covers both analytic black-box functions (BBOB) and data-defined problem
    instances (TSP, Knapsack, Graph Coloring). The key insight is that the
    environment carries the problem instance data and the genome is evaluated
    directly as a candidate solution — no MLP or program execution involved.

    Concrete subclasses must implement ``evaluate(solution)``.
    """

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a candidate solution vector against this problem instance.

        Args:
            solution: Raw solution vector from the genome (via IdentityInterpreter).

        Returns:
            Scalar objective value. Convention follows ``config.maximize``.
        """
        raise NotImplementedError


@struct.dataclass
class BaseSupervisedEnvironment(BaseEnvironment):
    """Environment for SupervisedTask: a static (X, y) dataset.

    Carries the full dataset as non-pytree fields so it remains static across
    ``jax.vmap`` lifting. Concrete subclasses populate ``data = (X, y)`` at
    construction time.

    The Interpreter is responsible for mapping ``genome → predictor(X) → y_hat``.
    The Evaluator computes the loss between ``y_hat`` and ``y``.
    """

    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    @property
    def X(self) -> chex.Array:
        """Input feature matrix, shape (n_samples, n_features)."""
        return self.data[0]

    @property
    def y(self) -> chex.Array:
        """Target labels/values, shape (n_samples,) or (n_samples, n_outputs)."""
        return self.data[1]


@struct.dataclass
class BaseRLEnvironment(BaseEnvironment):
    """Environment for ReinforcementTask: JAX-native RL interface.

    Normalizes the API across different RL libraries (gymnax, jumanji, brax)
    so that ``RLEvaluator`` never needs to know which library it is talking to.

    Concrete subclasses (``GymnaxEnv``, ``JumanjiEnv``) implement:
    - ``reset(key)`` → ``(obs, state)``
    - ``step(state, action, key)`` → ``(obs, state, reward, done)``
    - ``preprocess_obs(obs)`` → flat array (handles pytree observations)
    - ``postprocess_action(logits)`` → env-compatible action
    """

    def reset(self, key: chex.PRNGKey):
        """Reset the environment and return initial (obs, state)."""
        raise NotImplementedError

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey):
        """Take one step: (obs, state, reward, done, info)."""
        raise NotImplementedError

    def preprocess_obs(self, obs: Any) -> chex.Array:
        """Flatten/normalize raw environment observation to a 1D array."""
        raise NotImplementedError

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        """Convert interpreter output logits to environment-compatible action.

        Args:
            logits: Output array from the interpreter.
            raw_obs: The raw observation before preprocessing. Used by some
                environments (like Jumanji) to apply action masks.
        """
        raise NotImplementedError

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimensionality (required by MLPInterpreter)."""
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        """Action dimensionality (required by MLPInterpreter)."""
        raise NotImplementedError


# =============================================================================
# Output Modes
# =============================================================================

SUPPORTED_LOSS_FNS = ("mse", "bce", "mae")

@struct.dataclass
class BaseOutputMode:
    """Interface for evaluating a genome's result and computing fitness + info."""
    
    def process_sl(
        self, genome: Any, X: chex.Array, y: chex.Array, predictions: chex.Array
    ) -> tuple[chex.Array, dict]:
        raise NotImplementedError

    def process_opt(
        self, genome: Any, solution: chex.Array, raw_score: chex.Numeric
    ) -> tuple[chex.Array, dict]:
        raise NotImplementedError

    def process_rl(
        self, genome: Any, total_reward: chex.Numeric, final_state: Any, step_infos: Any
    ) -> tuple[chex.Array, dict]:
        raise NotImplementedError

    def observe_step(
        self, env_state: Any, obs: chex.Array, action: chex.Array, reward: chex.Numeric
    ) -> Any:
        """Called at each RL step to accumulate trajectory info (e.g. for QD)."""
        return None


@struct.dataclass
class ScalarOutput(BaseOutputMode):
    """Scalar fitness output mode for standard GA / ES engines.

    Computes a single scalar fitness per genome using the specified loss
    function. The ``maximize`` flag controls sign convention:
    - ``maximize=False`` (default): lower loss is better (minimization).
    - ``maximize=True``: higher value is better (maximization), loss is negated.
    """

    loss_fn: str = struct.field(pytree_node=False, default="mse")  # type: ignore[no-untyped-call]
    maximize: bool = struct.field(pytree_node=False, default=False)  # type: ignore[no-untyped-call]

    def compute_loss(self, predictions: chex.Array, targets: chex.Array) -> chex.Numeric:
        """Compute scalar loss between predictions and targets."""
        predictions = jnp.squeeze(predictions)

        if self.loss_fn == "mse":
            loss = jnp.mean(jnp.square(predictions - targets))
        elif self.loss_fn == "bce":
            probs = jax.nn.sigmoid(predictions)
            loss = -jnp.mean(
                targets * jnp.log(probs + 1e-7)
                + (1.0 - targets) * jnp.log(1.0 - probs + 1e-7)
            )
        elif self.loss_fn == "mae":
            loss = jnp.mean(jnp.abs(predictions - targets))
        else:
            raise ValueError(
                f"Unknown loss function '{self.loss_fn}'. "
                f"Supported: {SUPPORTED_LOSS_FNS}"
            )

        return self.format_fitness(loss)

    def format_fitness(self, score: chex.Numeric) -> chex.Numeric:
        """Format a raw scalar score according to the maximize flag.

        Since the core engine always minimizes, if maximize=True, we negate the score.
        """
        return -score if self.maximize else score

    # --- BaseOutputMode implementations ---

    def process_sl(self, genome: Any, X: chex.Array, y: chex.Array, predictions: chex.Array) -> tuple[chex.Array, dict]:
        return self.compute_loss(predictions, y), {}

    def process_opt(self, genome: Any, solution: chex.Array, raw_score: chex.Numeric) -> tuple[chex.Array, dict]:
        return self.format_fitness(raw_score), {}

    def process_rl(self, genome: Any, total_reward: chex.Numeric, final_state: Any, step_infos: Any) -> tuple[chex.Array, dict]:
        return self.format_fitness(total_reward), {}


@struct.dataclass
class BaseDescriptorFn:
    """Computes behavior descriptors for QD algorithms.

    Injected into ``QDOutput`` — the Evaluator itself knows nothing about
    descriptors. This keeps QD concerns fully orthogonal to the Task/Env/Interpreter
    decomposition.
    """

    def compute(self, genome: Any, inputs: chex.Array | None, outputs: chex.Array) -> chex.Array:
        """Compute behavior descriptor for Supervised/Optimization tasks."""
        raise NotImplementedError

    def compute_rl(self, genome: Any, final_state: Any, step_infos: Any) -> chex.Array:
        """Compute behavior descriptor for RL tasks."""
        raise NotImplementedError

    def observe_step(self, env_state: Any, obs: chex.Array, action: chex.Array, reward: chex.Numeric) -> Any:
        """Optional: Accumulate step info during RL rollouts."""
        return None


@struct.dataclass
class QDOutput(BaseOutputMode):
    """QD (Quality Diversity) output mode: fitness + behavior descriptors.

    Wraps a ``ScalarOutput`` for fitness and takes a separate
    ``BaseDescriptorFn`` for descriptor computation.
    """

    scalar_output: ScalarOutput
    descriptor_fn: BaseDescriptorFn = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    def process_sl(self, genome: Any, X: chex.Array, y: chex.Array, predictions: chex.Array) -> tuple[chex.Array, dict]:
        fitness, _ = self.scalar_output.process_sl(genome, X, y, predictions)
        desc = self.descriptor_fn.compute(genome, X, predictions)
        return fitness, {"descriptors": desc}

    def process_opt(self, genome: Any, solution: chex.Array, raw_score: chex.Numeric) -> tuple[chex.Array, dict]:
        fitness, _ = self.scalar_output.process_opt(genome, solution, raw_score)
        desc = self.descriptor_fn.compute(genome, None, solution)
        return fitness, {"descriptors": desc}

    def process_rl(self, genome: Any, total_reward: chex.Numeric, final_state: Any, step_infos: Any) -> tuple[chex.Array, dict]:
        fitness, _ = self.scalar_output.process_rl(genome, total_reward, final_state, step_infos)
        desc = self.descriptor_fn.compute_rl(genome, final_state, step_infos)
        return fitness, {"descriptors": desc}

    def observe_step(self, env_state: Any, obs: chex.Array, action: chex.Array, reward: chex.Numeric) -> Any:
        return self.descriptor_fn.observe_step(env_state, obs, action, reward)


@struct.dataclass
class MOOutput(BaseOutputMode):
    """Multi-Objective output mode: vector fitness per genome."""

    objectives: tuple[ScalarOutput, ...] = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    def process_sl(self, genome: Any, X: chex.Array, y: chex.Array, predictions: chex.Array) -> tuple[chex.Array, dict]:
        fitness_vec = jnp.stack([obj.compute_loss(predictions, y) for obj in self.objectives])
        return fitness_vec, {}

    def process_opt(self, genome: Any, solution: chex.Array, raw_score: chex.Numeric) -> tuple[chex.Array, dict]:
        fitness_vec = jnp.stack([obj.format_fitness(raw_score) for obj in self.objectives])
        return fitness_vec, {}

    def process_rl(self, genome: Any, total_reward: chex.Numeric, final_state: Any, step_infos: Any) -> tuple[chex.Array, dict]:
        fitness_vec = jnp.stack([obj.format_fitness(total_reward) for obj in self.objectives])
        return fitness_vec, {}
