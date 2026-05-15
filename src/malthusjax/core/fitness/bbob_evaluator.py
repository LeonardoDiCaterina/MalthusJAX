"""Wrapper around evosax BBOB benchmark problems.

Exposes a MalthusJAX-friendly evaluator that leverages the evosax
BBOBProblem API for deterministic black‑box function evaluation. Plays
nicely with our population and maximization conventions.
"""

# TODO: stop relying on evosax's internal problem registry
# use https://github.com/maxencefaldor/bbobax/tree/main/src/bbobax
# keep the evosax wrapper though to benchmark evosax implementation
# in their own benchmark suite

# # TODO: wrap also https://github.com/RobertTLange/gymnax
# in a different adapter for RL problems


from __future__ import annotations

from typing import Any, cast

import chex
import flax
import jax
from evosax.problems import BBOBProblem
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome

BBOB_NAME_ALIASES = {
    # Part 1: Separable functions
    "sphere": "sphere",
    "ellipsoidal": "ellipsoidal",
    "rastrigin": "rastrigin",
    "bueche_rastrigin": "bueche_rastrigin",
    "bueche-rastrigin": "bueche_rastrigin",
    "linear_slope": "linear_slope",
    "linear slope": "linear_slope",

    # Part 2: Functions with low or moderate conditioning
    "attractive_sector": "attractive_sector",
    "attractive sector": "attractive_sector",
    "step_ellipsoidal": "step_ellipsoidal",
    "step ellipsoidal": "step_ellipsoidal",
    "rosenbrock": "rosenbrock",
    "rosenbrock_original": "rosenbrock",
    "rosenbrock_rotated": "rosenbrock_rotated",

    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_original": "ellipsoidal",
    "ellipsoidal_rotated": "ellipsoidal_rotated",
    "discus": "discus",
    "bent_cigar": "bent_cigar",
    "bent cigar": "bent_cigar",
    "sharp_ridge": "sharp_ridge",
    "sharp ridge": "sharp_ridge",
    "different_powers": "different_powers",
    "different powers": "different_powers",

    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_original": "rastrigin",
    "rastrigin_rotated": "rastrigin_rotated",
    "weierstrass": "weierstrass",
    "schaffers_f7": "schaffers_f7",
    "schaffers f7": "schaffers_f7",
    "schaffers_f7_ill_cond": "schaffers_f7_ill_cond",
    "schaffers f7 ill-cond": "schaffers_f7_ill_cond",
    "griewank_rosenbrock": "griewank_rosenbrock",
    "griewank-rosenbrock": "griewank_rosenbrock",

    # Part 5: Multi-modal functions with weak global structure
    "schwefel": "schwefel",
    "gallagher_101_me": "gallagher_101_me",
    "gallagher 101-me": "gallagher_101_me",
    "gallagher_21_hi": "gallagher_21_hi",
    "gallagher 21-hi": "gallagher_21_hi",
    "katsuura": "katsuura",
    "lunacek": "lunacek",
}


@struct.dataclass
class BBOBConfig(BaseEvaluatorConfig):
    """Configuration for BBOB black-box optimization benchmarks.

    Attributes:
        fn_name: BBOB function identifier (lowercase: 'sphere', 'rastrigin', etc).
        num_dims: Problem dimensionality.
        seed: Seed ID for problem instance (rotation matrices, shifts).
    """

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore[no-untyped-call]
    num_dims: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    seed: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]


@struct.dataclass
class BBOBEvaluator(BaseEvaluator[RealGenome, BBOBConfig, Any]):
    """BBOB benchmark wrapper for MalthusJAX RealGenome optimization.

    Wraps evosax BBOBProblem for standard black-box test functions. The problem
    instance and its state are stored as non-PyTree fields (static).
    """

    # Evosax problem is static (doesn't change)
    evosax_problem: Any = flax.struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    problem_state: Any = flax.struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    @classmethod
    def create(cls, config: BBOBConfig) -> BBOBEvaluator:
        """Factory method to initialize the evosax BBOBProblem."""
        fn_name_lower = config.fn_name.lower()
        fn_name_normalized = BBOB_NAME_ALIASES.get(fn_name_lower, config.fn_name)

        problem = BBOBProblem(
            fn_name=fn_name_normalized,
            num_dims=config.num_dims,
            seed=config.seed,
        )

        rng = jax.random.PRNGKey(config.seed)
        problem_state = problem.init(rng)

        return cls(config=config, data=None, evosax_problem=problem, problem_state=problem_state)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Single genome evaluation via the evosax fitness function.

        The input vector is reshaped to match evosax expectations and evaluated
        deterministically. Returns raw minimization objective (lower=better).
        """
        x = genome.values[None, :]
        rng = jax.random.PRNGKey(0)

        fitness_scores, _, _ = self.evosax_problem.eval(rng, x, self.problem_state)
        result = fitness_scores[0]

        return result if self.config.maximize else -result

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """Vectorized batch evaluation using evosax native evaluation.

        The population genes array is fed directly into the fitness function with
        deterministic evaluation. Returns raw minimization objectives (lower=better).
        """
        X = population.genes.values

        rng = jax.random.PRNGKey(0)

        fitness_scores, _, _ = self.evosax_problem.eval(rng, X, self.problem_state)

        final_fitness = fitness_scores if self.config.maximize else -fitness_scores

        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=final_fitness)
        )

    @property
    def f_opt(self) -> chex.Numeric:
        """Reference minimum value (optimal function value) for this problem.

        This is the known global optimum from the BBOB benchmark suite.
        """
        return self.evosax_problem.f_opt

    @property
    def x_opt(self) -> chex.Array:
        """Reference optimal solution location for this problem.

        This is the known global optimum location from the BBOB benchmark suite.
        """
        return self.evosax_problem.x_opt

    def get_gap_to_optimum(self, fitness_value: chex.Numeric) -> chex.Numeric:
        """Compute gap between a fitness value and the known optimum.

        Args:
            fitness_value: The fitness value to compare (should be raw BBOB value,
                          not considering maximize flag).

        Returns:
            The absolute gap to the optimum (always positive, 0 means optimal).
        """
        return jax.numpy.abs(fitness_value - self.f_opt)
