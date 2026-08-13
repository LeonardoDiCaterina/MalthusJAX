#!/usr/bin/env python3
"""Community Showcase: Neuroevolution & Dynamic Topologies in Pure JAX.

Demonstrates how MalthusJAX orchestrates neuroevolution algorithms (TensorNEAT)
alongside custom pure-JAX dynamical control problems and hybrid MAP-Elites.

Zero external RL environment dependencies: everything runs natively in pure JAX!
"""

from __future__ import annotations

import pprint
import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp

try:
    import tensorneat
    from tensorneat.algorithm import NEAT
    from tensorneat.common import State
    from tensorneat.genome import DefaultGenome
    from tensorneat.problem import BaseProblem

    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False
    BaseProblem = object  # Fallback for typing

from malthusjax.composer import Composer
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.core.fitness.qd.tensorneat_evaluator import (
    TensorNeatEvaluatorConfig,
    TensorNeatQDEvaluator,
)
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter


# ===========================================================================
# 1. Custom Pure-JAX Quality-Diversity Pendulum Problem
# ===========================================================================
class CustomJAXQDPendulumProblem(BaseProblem):
    """Custom pure-JAX Quality-Diversity Inverted Pendulum environment.

    Simulates 25 time-steps of an inverted pendulum system. The neural network
    policy receives state [theta, theta_dot] and outputs control torque u.
    Returns both cumulative fitness AND 2D behavioral descriptors [final_theta, final_theta_dot].
    """

    jittable = True

    def __init__(self, dt: float = 0.05, steps: int = 25):
        self.dt = dt
        self.steps = steps

    @property
    def num_inputs(self) -> int:
        return 2  # [angle theta, angular velocity theta_dot]

    @property
    def num_outputs(self) -> int:
        return 1  # Control torque u

    def evaluate(
        self, state: State, key: chex.PRNGKey, forward_fn: Any, individual: Any
    ) -> Tuple[chex.Numeric, chex.Array]:
        """Simulate pendulum dynamics and return (fitness, 2D behavioral descriptors)."""

        def step_fn(carry_state: Tuple[chex.Array, chex.Array], _: Any):
            theta, theta_dot = carry_state
            obs = jnp.array([theta, theta_dot])

            # Forward pass through the evolved TensorNEAT neural network
            action = forward_fn(state, individual, obs)[0]
            u = jnp.clip(action, -2.0, 2.0)

            # Physics update: d^2(theta)/dt^2 = g*sin(theta) - b*theta_dot + u
            g, b = 9.81, 0.1
            theta_acc = g * jnp.sin(theta) - b * theta_dot + u
            new_theta_dot = theta_dot + theta_acc * self.dt
            new_theta = theta + new_theta_dot * self.dt

            # Loss penalty: angle deviation + velocity + control effort
            step_cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2
            return (new_theta, new_theta_dot), step_cost

        # Initial state: small perturbation from vertical upright position (theta=0.2 rad)
        init_state = (jnp.array(0.2), jnp.array(0.0))
        (final_theta, final_theta_dot), step_costs = jax.lax.scan(
            step_fn, init_state, None, length=self.steps
        )

        # Return fitness and 2D behavioral descriptors (final physical state)
        fitness = -jnp.sum(step_costs)
        descriptors = jnp.array([final_theta, final_theta_dot])
        return fitness, descriptors


# ===========================================================================
# 2. Main Community Showcase Harness
# ===========================================================================
def main() -> None:
    if not TENSORNEAT_AVAILABLE:
        print(
            "TensorNEAT is not installed. Please run `pip install tensorneat` to run this showcase."
        )
        return

    print("=" * 75)
    print("  MALTHUSJAX COMMUNITY SHOWCASE: NEUROEVOLUTION IN PURE JAX")
    print("=" * 75)

    print("\n[Step 1] Creating Custom Pure-JAX Pendulum Control Problem...")
    problem = CustomJAXQDPendulumProblem(steps=25)
    print(
        f"         Problem: Inverted Pendulum (Inputs={problem.num_inputs}, Outputs={problem.num_outputs})"
    )

    pop_size = 128
    generations = 50
    seeds = (42, 43)

    print("\n[Step 2] Defining Neuroevolution Pipelines:")
    print("         1. TensorNEAT (NEAT)               -> Standard Dynamic Graph Evolution")
    print("         2. TensorNEAT (SubNEAT)            -> Subspecies Topology Search")
    print(
        "         3. MalthusJAX Native (MAP-Elites)   -> TensorNeatEmitter in MalthusJAX MAP-Elites"
    )

    composer = Composer.create_default()

    # Build MalthusJAX Native TensorNEAT Emitter
    tn_genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)
    tn_algorithm = NEAT(pop_size=pop_size, species_size=2, genome=tn_genome)

    native_evaluator = TensorNeatQDEvaluator.create(
        algorithm=tn_algorithm,
        problem=problem,
        forward_fn=tn_algorithm.forward,
        config=TensorNeatEvaluatorConfig(seed=42, maximize=True),
    )
    native_emitter = TensorNeatEmitter(_batch_size=pop_size, genome=tn_genome)

    # Define Pipelines
    pipelines = {
        "TensorNEAT (NEAT)": {
            "backend": "tensorneat",
            "fitness": problem,
            "strategy": f"tensorneat:algorithm=neat:pop_size={pop_size}:species_size=2:max_nodes=10:max_conns=20",
            "eval_mode": "native",
        },
        "TensorNEAT (SubNEAT)": {
            "backend": "tensorneat",
            "fitness": problem,
            "strategy": f"tensorneat:algorithm=subneat:pop_size={pop_size}:species_size=2:max_nodes=10:max_conns=20",
            "eval_mode": "native",
        },
        "MalthusJAX Native (TensorNEAT MAP-Elites)": {
            "strategy": MapElitesStrategy(
                emitter=native_emitter, num_centroids=pop_size, num_descriptors=2
            ),
            "fitness": native_evaluator,
        },
    }

    print("\n[Step 3] Executing Parallel Neuroevolution Benchmarks:")
    print(f"         - Population Size: {pop_size}")
    print(f"         - Generations:     {generations}")
    print(f"         - Random Seeds:    {seeds}\n")

    t0 = time.time()
    comparison = composer.compare(
        pipelines=pipelines,
        pop_size=pop_size,
        generations=generations,
        seeds=seeds,
        shared_initial_population=False,
        bounds=(-2.0, 2.0),
    )
    wall_time = time.time() - t0

    print("\n[Step 4] Showcase Execution Finished! Results Summary:\n")
    summary = comparison.summary_table()
    pprint.pprint(summary)

    print(f"\nTotal Elapsed Time: {wall_time:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
