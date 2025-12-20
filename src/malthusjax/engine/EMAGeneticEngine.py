from flax import struct
import jax.numpy as jnp
import chex
from typing import Any

# Import existing components
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEvolutionState, GeneticEngineParams

@struct.dataclass
class EMAGeneticEngineParams(GeneticEngineParams):
    """
    Parameters for the EMA Genetic Engine.
    Inherits all standard GeneticEngineParams.
    """
    ema_alpha: float = 0.1       # Decay rate for EMA (0.1 = ~10 gen memory)
    target_ema: float = 0.2      # Target EMA value (e.g., 20% success rate)
    sensitivity: float = 1.0     # Sensitivity of correction to error

# 1. EXTEND THE STATE
# We just inherit and add the one field we need.
@struct.dataclass
class EMAGeneticState(GeneticEvolutionState):
    progress_ema: float = 0.0  # Exponential Moving Average of success/delta

# 2. EXTEND THE ENGINE
@struct.dataclass
class EMAGeneticEngine(GeneticEngine):
    """
    Adaptive Genetic Engine using EMA control for mutation strength.
    """
    # --- A. CUSTOMIZABLE LOGIC ---
    def _compute_delta(self, old_state, new_state) -> float:
        """
        Metric to smooth. 
        Default: 1.0 if Best Fitness improved, 0.0 otherwise.
        """
        # Using .astype(float) to make it math-compatible
        return (new_state.best_fitness > old_state.best_fitness).astype(jnp.float32)

    def _correction_fn(self, ema_value: float) -> float:
        """
        PID Control Law:
        If EMA > Target (Too Easy) -> Grow Sigma (>1.0)
        If EMA < Target (Too Hard) -> Shrink Sigma (<1.0)
        """
        # Ex: exp(1.0 * (0.25 - 0.20)) = exp(0.05) ≈ 1.05 (Grow by 5%)
        # Ex: exp(1.0 * (0.10 - 0.20)) = exp(-0.1) ≈ 0.90 (Shrink by 10%)
        error = ema_value - self.engine_params.target_ema
        return jnp.exp(self.engine_params.sensitivity * error)

    # --- B. OVERRIDE INIT ---
    def init_state(self, rng_key: chex.Array) -> EMAGeneticState:
        # Get the standard state
        base = super().init_state(rng_key)
        
        # Promote it to EMA state
        return EMAGeneticState(
            **base.__dict__,
            progress_ema=0.0 # Start with 0 momentum
        )

    # --- C. OVERRIDE STEP ---
    def step(self, state: EMAGeneticState):
        # 1. Run the Standard Step (Reuse all your optimized code)
        new_state, metrics = super().step(state)
        
        # 2. Calculate Signal
        delta = self._compute_delta(state, new_state)
        
        # 3. Update EMA (The Integrator)
        new_ema = (self.engine_params.ema_alpha * delta) + ((1.0 - self.engine_params.ema_alpha) * state.progress_ema)
        
        # 4. Calculate Correction
        correction = self._correction_fn(new_ema)
        
        # 5. Apply to Mutation Operator
        # We assume the operator has 'mutation_strength' (like GaussianMutation)
        old_op = new_state.operators.mutation
        
        # Update and Clip for numerical stability
        new_sigma = jnp.clip(
            old_op.mutation_strength * correction, 
            1e-6, 
            100.0
        )
        
        # 6. Re-Bake Operator into State
        new_op = old_op.replace(mutation_strength=new_sigma)
        new_operators = new_state.operators.replace(mutation=new_op)
        
        # 7. Return Extended State
        final_state = new_state.replace(
            operators=new_operators,
            progress_ema=new_ema
        )
        
        return final_state, metrics