from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
import jax
import jax.numpy as jnp

# --- MalthusJAX Imports ---
from malthusjax.engine.genetic_fastengine import GeneticEngine

# --- Evosax Imports ---
from evosax.problems.problem import Problem as EvosaxProblem

class AbstractBenchmarkAdapter(ABC):
    """
    The Universal Plug. 
    Guarantees that any engine provides a 'step' function compatible with jax.lax.scan.
    """
    
    @abstractmethod
    def init(self, rng: jax.Array) -> Any:
        """Returns the initial carry state for the loop."""
        pass

    @abstractmethod
    def make_step_fn(self) -> Callable:
        """Returns (carry, _) -> (new_carry, metrics_scalar)."""
        pass
    
    @abstractmethod
    def get_best_fitness(self, carry: Any) -> float:
        """Extracts scalar best fitness."""
        pass

    @abstractmethod
    def get_device_info(self) -> str:
        """Returns the JAX device name (e.g., 'NVIDIA A100-SXM4-40GB')."""
        pass


class MalthusAdapter(AbstractBenchmarkAdapter):
    def __init__(self, engine: GeneticEngine):
        self.engine = engine

    def init(self, rng: jax.Array) -> Any:
        return self.engine.init_state(rng)

    def make_step_fn(self) -> Callable:
        def step(carry, _):
            state = carry
            new_state, metrics = self.engine.step(state)
            return new_state, metrics.best_fitness
        return step

    def get_best_fitness(self, carry: Any) -> float:
        # Assuming Malthus Engine handles direction (max/min) internally
        return float(carry.best_fitness)

    def get_device_info(self) -> str:
        # Check the device of the initial state if available, else default
        try:
            return jax.devices()[0].device_kind
        except Exception:
            return "Unknown Device"


class EvosaxAdapter(AbstractBenchmarkAdapter):
    def __init__(self, strategy: Any, params: Any, problem: EvosaxProblem):
        self.strategy = strategy
        self.es_params = params
        self.problem = problem

    def init(self, rng: jax.Array) -> Any:
        # Prepare RNGs
        rng, rng_init = jax.random.split(rng, 2)

        # Infer population size from the strategy
        pop_size = getattr(self.strategy, 'population_size', None) or getattr(self.strategy, 'pop_size', None)

        # Infer problem dimensionality
        num_dims = getattr(self.problem, 'num_dims', None) or getattr(self.problem, 'dimension', None)

        # Build a simple initial population and placeholder fitness array.
        # This mirrors the approach used in the repo's demos: uniform samples in [-5, 5].
        initial_pop = jax.random.uniform(rng_init, (pop_size, num_dims), minval=-5.0, maxval=5.0)
        initial_fitness = jnp.full((pop_size,), jnp.inf)

        # Initialize strategy and problem states using the Evosax API: init(rng, pop, fitness, params)
        state = self.strategy.init(rng, initial_pop, initial_fitness, self.es_params)

        # Initialize problem-specific parameters/state. Use a fixed key if the problem expects deterministic setup,
        # otherwise pass a split key derived from rng.
        problem_state = self.problem.init(rng)

        return (state, problem_state, rng)

    def make_step_fn(self) -> Callable:
        strategy, problem, es_params = self.strategy, self.problem, self.es_params
        
        def step(carry, _):
            state, param_state, rng = carry
            rng, rng_ask, rng_eval, rng_tell = jax.random.split(rng, 4)
            x, state = strategy.ask(rng_ask, state, es_params)
            fitness, new_param_state, _ = problem.eval(rng_eval, x, param_state)
            state, _ = strategy.tell(rng_tell, x, fitness, state, es_params)
            return (state, new_param_state, rng), state.best_fitness
        return step

    def get_best_fitness(self, carry: Any) -> float:
        state, _, _ = carry
        return float(state.best_fitness)

    def get_device_info(self) -> str:
        return jax.devices()[0].device_kind