from flax import struct
import jax
import jax.numpy as jnp
import chex
from malthusjax.operators.base import BaseSelection

@struct.dataclass
class RouletteSelection(BaseSelection):
    """
    Selection operator that samples parents proportional to their fitness.
    
    Optimizations:
    - If num_selections == pop_size: Uses Gumbel-Max trick (fully parallel, no sort/scan).
    - Otherwise: Uses jax.random.choice (standard inverse transform sampling).
    """
    temperature: float = 1.0
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, population) -> chex.Array:
        """
        Args:
            keys: Random keys for sampling.
            fitness: Score array (Higher is better).
        """
        rng = keys[0]
        pop_size = fitness.shape[0]
        
        # 1. Compute Logits (Numerical Stability)
        # Shift logits to avoid overflow in exp()
        logits = fitness / self.temperature
        logits = logits - jnp.max(logits)
        
        # 2. Branching Logic
        # On GPU, JAX traces both branches, so this 'if' is evaluated at trace time.
        # Since self.num_selections and pop_size are static integers, 
        # only the correct branch is compiled.
        
        if self.num_selections == pop_size:
            # === OPTIMIZED PATH: Gumbel-Max Trick ===
            # This generates N samples in O(1) parallel time without sorting or prefix sums.
            # Logic: argmax(logits + gumbel_noise) is equivalent to categorical sampling.
            
            # We need to sample 'num_selections' times. 
            # Since num_selections == pop_size, we generate a noise matrix of shape (N, N)
            # where each row 'i' represents one independent sampling event.
            # Note: This consumes O(N^2) memory for noise, which is fine for N=128-4096.
            # For massive N (>16k), this might OOM, but it's the fastest method.
            
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, shape=(self.num_selections, pop_size))))
            selected_indices = jnp.argmax(logits + gumbel_noise, axis=1)
            
        else:
            # === STANDARD PATH: jax.random.choice ===
            # Standard implementation using searchsorted (Roulette).
            # Better for when we need few samples (k << N) or weird shapes.
            
            exp_vals = jnp.exp(logits)
            probs = exp_vals / jnp.sum(exp_vals)
            
            selected_indices = jax.random.choice(
                rng, 
                a=pop_size, 
                shape=(self.num_selections,), 
                p=probs, 
                replace=True
            )
            
        return selected_indices