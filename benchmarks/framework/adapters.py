import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, Any, Callable

# --- MalthusJAX Imports ---
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
# --- Evosax Imports ---
from evosax.problems import BBOBProblem

# ==============================================================================
# 1. HELPER: Problem Instantiation
# ==============================================================================

def setup_bbob_instances(problem_name: str, dim: int, seed: int):
    """
    Creates the objective function for both frameworks.
    """
    # Malthus uses Maximization by default, so we flip BBOB (min) to max
    # This ensures both frameworks are solving the same math, just signed differently
    bbob_config = BBOBConfig(fn_name=problem_name, num_dims=dim, seed=seed, maximize=True)
    mjx_evaluator = BBOBEvaluator.create(bbob_config)
    
    # EvoSax - minimization by default
    es_problem = BBOBProblem(problem_name, num_dims=dim, seed=seed)
    
    return mjx_evaluator, es_problem

# ==============================================================================
# 2. ABSTRACT ADAPTERS
# ==============================================================================

class AbstractBenchmarkAdapter(ABC):
    """
    Unified interface for benchmarking different frameworks.
    Guarantees that the Runner sees the same API regardless of backend.
    """
    @abstractmethod
    def init(self, rng: jax.Array) -> Any:
        pass

    @abstractmethod
    def make_step_fn(self) -> Callable:
        """Returns a function: step(carry) -> (new_carry, metrics)"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> str:
        pass

# ==============================================================================
# 3. CONCRETE ADAPTERS
# ==============================================================================

class MalthusJaxAdapter(AbstractBenchmarkAdapter):
    def __init__(self, engine):
        self.engine = engine
    
    def init(self, rng):
        return self.engine.init_state(rng)
    
    def make_step_fn(self):
        # MalthusJAX engine.step returns (state, metrics)
        # The runner expects exactly this signature.
        return self.engine.step
        
    def get_device_info(self):
        return str(jax.devices()[0].device_kind)

class EvosaxAdapter(AbstractBenchmarkAdapter):
    def __init__(self, strategy, params, problem):
        self.strategy = strategy
        self.params = params
        self.problem = problem
        
    def init(self, rng):
        r_init, r_start = jax.random.split(rng)
        
        # 1. Init Problem State
        p_state = self.problem.init(r_init)
        
        # 2. Init Strategy State
        # Evosax usually needs an initial solution to shape the state
        # We sample one randomly just for shape inference
        init_x = self.problem.sample(r_init)
        
        # Note: Some Evosax strats need 'init_fitness' too
        # We'll do a dummy eval to get it
        init_fit, p_state, _ = self.problem.eval(r_init, init_x, p_state)
        
        state = self.strategy.init(r_init, self.problem.sample(r_init), init_fit, self.params)
        
        return (state, p_state, r_start)

    def make_step_fn(self):
        # We must capture 'strategy', 'params', 'problem' in closure
        strategy = self.strategy
        params = self.params
        problem = self.problem
        
        def step(carry, _=None):
            state, p_state, rng = carry
            rng, rng_step = jax.random.split(rng)
            
            # 1. ASK
            x, state = strategy.ask(rng_step, state, params)
            
            # 2. EVAL
            # EvoSax returns (fitness, new_p_state, metrics)
            fitness, p_state, _ = problem.eval(rng_step, x, p_state)
            
            # 3. TELL
            state, _ = strategy.tell(rng_step, x, fitness, state, params)
            
            # Return matching Malthus signature: (new_carry, metrics)
            # We return None for metrics to keep overhead minimal
            return (state, p_state, rng), None
            
        return step

    def get_device_info(self):
        return str(jax.devices()[0].device_kind)