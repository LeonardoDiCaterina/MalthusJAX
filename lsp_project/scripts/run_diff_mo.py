"""Runner for Differentiable Multi-Objective Symbolic Regression."""

import argparse
import time

import jax
import jax.numpy as jnp
from lsp.evaluator.differentiable_evaluator import (
    DifferentiableLinearGPEvaluatorConfig,
    DifferentiableMOEvaluator,
)
from lsp.evaluator.operator_set import DEFAULT_DIFFERENTIABLE_OPS
from lsp.genome import PrefixGenomeConfig
from lsp.operators.crossover import HomologousPrefixCrossover
from lsp.operators.emitter import LSPMOEmitter
from lsp.operators.mutation import AnnealedTopologicalMutation

from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=256)
    parser.add_argument("--length", type=int, default=16)
    return parser.parse_args()


def generate_toy_physics_data(key, n_samples=50):
    """Generate X, Y, dY_dX for a synthetic physics problem.

    True function: f(x0, x1) = x0 * x1 + sin(x0)
    Gradients:
      df/dx0 = x1 + cos(x0)
      df/dx1 = x0
    """
    X = jax.random.uniform(key, shape=(n_samples, 2), minval=-2.0, maxval=2.0)

    x0 = X[:, 0]
    x1 = X[:, 1]

    # Target values
    Y = x0 * x1 + jnp.sin(x0)

    # Target gradients (Jacobian)
    dy_dx0 = x1 + jnp.cos(x0)
    dy_dx1 = x0
    dY_dX = jnp.stack([dy_dx0, dy_dx1], axis=-1)  # shape (n_samples, 2)

    return X, Y, dY_dX


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    k_data, k_init, k_state = jax.random.split(key, 3)

    # 1. Get Data
    X_train, Y_train, dY_dX_train = generate_toy_physics_data(k_data)
    num_inputs = X_train.shape[1]

    # 2. Configure Genome
    num_ops = len(DEFAULT_DIFFERENTIABLE_OPS.op_names)
    genome_config = PrefixGenomeConfig(
        length=args.length,
        num_inputs=num_inputs,
        num_ops=num_ops,
        max_arity=3,
    )

    # 3. Configure Differentiable MO Evaluator
    evaluator_config = DifferentiableLinearGPEvaluatorConfig(
        num_inputs=num_inputs,
        length=args.length,
        grad_weight=1.0,  # Used internally for best row selection
    )
    evaluator = DifferentiableMOEvaluator(
        config=evaluator_config, data=(X_train, Y_train, dY_dX_train)
    )

    # 4. Setup Operators & Emitter
    mutation = AnnealedTopologicalMutation(
        op_rate=0.1, arg_rate=0.2, p_input_start=0.1, p_input_end=0.9
    )
    crossover = HomologousPrefixCrossover()

    emitter = LSPMOEmitter(
        mutation=mutation,
        crossover=crossover,
        variation_percentage=0.5,
        genome_config=genome_config,
        _batch_size=args.pop_size,
    )

    engine = MOEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=MOEngineParams(pop_size=args.pop_size, num_generations=args.generations),
    )

    # 5. Initialize State
    initial_pop = genome_config.init_population(k_init, args.pop_size)
    state = engine.init_state(k_state, initial_pop)

    step_fn = jax.jit(engine.step)

    print(f"Starting MO Symbolic AD | pop={args.pop_size} | gens={args.generations}")
    start_time = time.time()

    for gen in range(args.generations):
        state, kpi = step_fn(state)

        num_pareto = int(kpi.num_pareto_optimal)

        if (gen + 1) % 10 == 0:
            print(f"Gen {gen + 1}/{args.generations} | Pareto count: {num_pareto}")

    # Extract Pareto Front
    final_pop = state.population
    pareto_mask = final_pop.pareto_rank == 0
    pareto_fitness = final_pop.fitness[pareto_mask]  # (N_pareto, 3)

    print(f"\nDone in {time.time() - start_time:.2f}s.")
    print(f"Found {len(pareto_fitness)} models on the Pareto Front.")
    print("Top models (sorted by Value MSE):")

    # Sort by Value MSE
    sorted_indices = jnp.argsort(pareto_fitness[:, 0])
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        v_mse = pareto_fitness[idx, 0]
        g_mse = pareto_fitness[idx, 1]
        nodes = pareto_fitness[idx, 2]
        print(f"  [{i + 1}] Val MSE: {v_mse:.6f} | Grad MSE: {g_mse:.6f} | Nodes: {nodes}")


if __name__ == "__main__":
    main()
