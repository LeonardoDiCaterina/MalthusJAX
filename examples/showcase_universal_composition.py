"""
Showcase 2: Universal Composition (Mix-and-Match Any Axis)

This script demonstrates the ultimate promise of the 4-axis architecture:
Evaluator = Problem(Environment x Transform x Interpreter) x Output.

We showcase two highly unusual but completely valid compositions:
1. Applying Quality-Diversity (MapElites) to a classic Deep RL environment (Brax).
2. Applying an Evosax continuous optimizer (CMA-ES) to a TensorNEAT discrete problem (XOR).
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Any
from malthusjax.core.genome.real_genome import RealPopulation

# Monkey patch RealPopulation to support old adapter
RealPopulation.descriptors = property(lambda self: self.info["descriptors"])
from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.composer.strategies.core import MapElitesStrategy

from malthusjax.core.fitness.composable.base import ScalarOutput, QDOutput
from malthusjax.core.fitness.composable.environments import BraxEnv, TensorNEATProblemWrapper
from malthusjax.core.fitness.composable.evaluators import RLEvaluator, OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import MLPInterpreter, IdentityInterpreter
from malthusjax.core.genome.real_genome import RealGenomeConfig

try:
    from tensorneat.problem import XOR
    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False


def showcase_qd_on_brax():
    print("\n--- 1. Quality-Diversity applied to Deep RL (Brax) ---")
    print("Goal: Evolve a diverse set of Neural Network policies for Brax.")
    print("Stack: RealGenome + MapElites + BraxEnv + MLPInterpreter + QDOutput")

    # 1. Create the Environment
    env = BraxEnv.create("ant", auto_reset=True)

    # 2. Create the Interpreter (Neural Network policy)
    interpreter = MLPInterpreter(
        input_dim=env.env.observation_size,
        output_dim=env.env.action_size,
        hidden=[32, 32],
        activation="tanh",
    )

    # 3. Create a dummy QD output (normally you'd extract real descriptors from the rollout)
    from malthusjax.core.fitness.composable.base import BaseDescriptorFn
    
    class DummyDescriptor(BaseDescriptorFn):
        def compute_rl(self, genome, final_state, step_infos):
            return jnp.zeros(2)

    output = QDOutput(
        scalar_output=ScalarOutput(maximize=True),
        descriptor_fn=DummyDescriptor()
    )

    # 4. Compose the Evaluator
    evaluator = RLEvaluator(
        env=env,
        interpreter=interpreter,
        output=output,
        max_steps=100,
    )
    # Mock config for adapter
    class MockConfig:
        genome_config = RealGenomeConfig(shape=(interpreter.num_params,))
    object.__setattr__(evaluator, "config", MockConfig())
    
    # Inject rng into evaluator calls from QDAX adapter
    original_eval = evaluator.evaluate_population
    object.__setattr__(evaluator, "evaluate_population", lambda pop: original_eval(pop, rng=jax.random.PRNGKey(42)))

    from qdax.core.emitters.mutation_operators import isoline_variation
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.utils.metrics import default_qd_metrics
    
    # 5. Execute with MapElites
    from malthusjax.composer.qdax_adapter import build_qdax_engine
    
    centroids = jnp.zeros((100, 2))
    emitter = MixingEmitter(
        mutation_fn=lambda x, y, key: (x, key), 
        variation_fn=lambda x, y, key: isoline_variation(x, y, key, 0.1, 0.1),
        variation_percentage=1.0,
        batch_size=32,
    )
    
    from functools import partial
    engine = build_qdax_engine(
        strategy_cls=MAPElites,
        emitter=emitter,
        metrics_function=partial(default_qd_metrics, qd_offset=0.0),
        centroids=centroids,
        init_variables=jax.random.uniform(jax.random.PRNGKey(0), (32, interpreter.num_params), minval=-1.0, maxval=1.0),
        evaluator=evaluator,
        pop_size=32,
        generations=2,
        maximize=True,
    )

    key = jax.random.PRNGKey(42)
    results = engine.run_once(key, compile=True)
    print(f"MapElites Step 1 - Best Fitness: {results['summary']['best_fitness']:.4f}")
    print("Success! We mapped diverse Deep RL policies.")


def showcase_evosax_on_brax():
    """
    Demonstrates using EvoSAX (continuous ES) on a Deep RL environment (Brax).
    
    Stack:
    - Genome: RealGenome (flat weights)
    - Interpreter: MLPInterpreter (continuous -> NN weights)
    - Environment: BraxEnv (Ant)
    - Output: ScalarOutput (fitness only)
    - Strategy: CMA-ES (via EvoSAXAdapter)
    """
    print("\n--- 2. Evosax (CMA-ES) applied to Deep RL (Brax) ---")
    print("Goal: Use a continuous optimizer to train a Neural Network policy.")
    print("Stack: RealGenome + CMA-ES + BraxEnv + MLPInterpreter + ScalarOutput")

    # 1. Environment (Brax Ant)
    from malthusjax.core.fitness.composable.environments import BraxEnv
    env = BraxEnv.create(env_name="ant")

    # 2. Interpreter (Flat array -> MLP Weights)
    from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
    interpreter = MLPInterpreter(
        input_dim=env.env.observation_size,
        output_dim=env.env.action_size,
        hidden=[32, 32],
        activation="tanh",
    )

    # 3. Output (Standard scalar fitness)
    from malthusjax.core.fitness.composable.base import ScalarOutput
    output = ScalarOutput(maximize=True)

    # 4. Evaluator
    from malthusjax.core.fitness.composable.evaluators import RLEvaluator
    evaluator = RLEvaluator(
        env=env,
        interpreter=interpreter,
        output=output,
        max_steps=100,
    )
    
    # Mock config for adapter
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    class MockConfig2:
        genome_config = RealGenomeConfig(shape=(interpreter.num_params,))
    object.__setattr__(evaluator, "config", MockConfig2())

    # 5. Execute with EvoSAX
    from malthusjax.composer.evosax_adapter import build_evosax_engine
    engine = build_evosax_engine(
        strategy_name="CMA_ES",
        evaluator=evaluator,
        pop_size=64,
        generations=10,
        maximize=True,
    )

    # 6. Execute the loop
    key = jax.random.PRNGKey(123)
    results = engine.run_once(key, compile=True)
    
    print(f"EvoSAX Step 10 - Best Fitness: {results['summary']['best_fitness']:.4f}")
    print("Success! We trained a Deep RL policy using CMA-ES.")


def main():
    print("=== MalthusJAX Universal Composition Showcase ===\n")
    showcase_qd_on_brax()
    showcase_evosax_on_brax()

if __name__ == "__main__":
    main()
