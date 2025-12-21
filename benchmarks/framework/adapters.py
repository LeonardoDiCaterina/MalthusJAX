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
    bbob_config = BBOBConfig(fn_name=problem_name, num_dims=dim, seed=seed, maximize=True)
    mjx_evaluator = BBOBEvaluator.create(bbob_config)
    
    # EvoSax - minimization by default
    es_problem = BBOBProblem(problem_name, num_dims=dim, seed=seed)
    
    return mjx_evaluator, es_problem

# ==============================================================================
# 2. ABSTRACT ADAPTERS
# ==============================================================================

class AbstractBenchmarkAdapter(ABC):
    @abstractmethod
    def init(self, rng: jax.Array) -> Any:
        pass

    @abstractmethod
    def make_step_fn(self) -> Callable:
        pass
    
    @abstractmethod
    def get_device_info(self) -> str:
        pass

# ==============================================================================
# 3. CONCRETE ADAPTERS
# ==============================================================================

class MalthusAdapter(AbstractBenchmarkAdapter):
    def __init__(self, engine):
        self.engine = engine
    
    def init(self, rng):
        return self.engine.init_state(rng)
    
    def make_step_fn(self):
        # FIX: Wrap the engine step to handle the 'x' argument from scan
        def step(carry, _):
            return self.engine.step(carry)
        return step
        
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
        # We sample a prototype to let Evosax infer shapes
        init_x = self.problem.sample(r_init)
        
        # FIX: Do NOT call eval here. Just use infinity (worst case for Min).
        # This avoids the shape crash in BBOB Sphere.
        init_fit = jnp.array(float('inf')) 
        
        state = self.strategy.init(r_init, init_x, init_fit, self.params)
        
        return (state, p_state, r_start)

    def make_step_fn(self):
        strategy = self.strategy
        params = self.params
        problem = self.problem
        
        def step(carry, _=None):
            state, p_state, rng = carry
            rng, rng_step = jax.random.split(rng)
            
            # 1. ASK
            x, state = strategy.ask(rng_step, state, params)
            
            # 2. EVAL
            fitness, p_state, _ = problem.eval(rng_step, x, p_state)
            
            # 3. TELL
            state, _ = strategy.tell(rng_step, x, fitness, state, params)
            
            return (state, p_state, rng), None
            
        return step

    def get_device_info(self):
        return str(jax.devices()[0].device_kind)